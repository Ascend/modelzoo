# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders_8p import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import torch.autograd.profiler as prof

import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

#CALCULATE_DEVICE = "npu:4"
#PRINT_DEVICE = "cpu"
#torch.npu.set_device(CALCULATE_DEVICE)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # Define network
        print("class: {} backbone: {} output_stride: {} sync_bn: {} freeze_bn: {}".format(self.nclass,args.backbone,args.out_stride,args.sync_bn,args.freeze_bn))
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,device=torch.device(args.CALCULATE_DEVICE))
        #print('model parameter: {}\n'.format(model.summary()))
        #torch.save(model, "./saved_model.pth")
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(size_average=True,ignore_index=255).to(torch.device(args.CALCULATE_DEVICE))
        self.model, self.optimizer = model, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        self.model.to(args.CALCULATE_DEVICE)
        
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch,args):
        train_loss = 0.0
        self.model.train()        
        self.model = self.model.to(torch.device(args.CALCULATE_DEVICE))
        tbar = tqdm(self.train_loader)
        print('suspect')
        num_img_tr = len(self.train_loader)
        pre_time = 0
        core_time = 0
        itr = 0
        for i, sample in enumerate(tbar):
          t_int = time.time()
          image, target = sample['image'], sample['label']
          image, target = image.to(args.CALCULATE_DEVICE, non_blocking=True), target.to(args.CALCULATE_DEVICE, non_blocking=True)
          self.scheduler(self.optimizer, i, epoch, self.best_pred)
          self.optimizer.zero_grad()
          output = self.model(image)
          output = output.permute(0, 2, 3, 1).reshape(-1,21)
          target = target.flatten().int()
          loss = self.criterion(output, target)
          loss.backward()
          self.optimizer.step()
          train_loss += loss.item()
          tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
          print('Train loss: %.3f' % (train_loss / (i + 1)))
          self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            

          self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
          print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
      
    def validation(self, epoch,args):
        self.model.eval()
        self.evaluator.reset()
        #self.model.aspp.is_train = False
        #self.model.decoder.is_train = False
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            with torch.no_grad():
                output = self.model(image.to(args.CALCULATE_DEVICE))



            output = output.permute(0, 2, 3, 1)
            output = torch.reshape(output, (-1,21))
            target = target.flatten().int()
            loss = self.criterion(output.to(args.CALCULATE_DEVICE), target.to(args.CALCULATE_DEVICE))
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        #if new_pred > self.best_pred:
        #    is_best = True
        #    self.best_pred = new_pred
        #    self.saver.save_checkpoint({
        #        'epoch': epoch + 1,
        #        'state_dict': self.model.module.state_dict(),
        #        'optimizer': self.optimizer.state_dict(),
        #        'best_pred': self.best_pred,
        #    }, is_best)
def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--npu', default=None, type=int, help='NPU id to use.')
    #parser.add_argument('--seed', type=int, default=123456, help='random seed')
    #parser.add_argument('--amp_cfg', action='store_true', help='If true, use'
    #                                                           'apex.')
    #parser.add_argument('--opt_level', default='O0', type=str,
    #                    help='set opt level.')
    #parser.add_argument('--loss_scale_value', default=1024, type=int,
    #                    help='set loss scale value.')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
    #parser.add_argument('--batch_size', default=64, type=int,
    #                    help='set batch_size')
    #parser.add_argument('--epochs', default=20, type=int, help='set epochs')
    #parser.add_argument('--epochs_per_save', default=1, type=int,
    #                    help='save per epoch')
    #parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                    metavar='LR', help='initial learning rate', dest='lr')
    #parser.add_argument('--workers', default=0, type=int, help='set workers')
    #parser.add_argument('--data_dir', default="", type=str,
    #                    help='set data_dir')
    parser.add_argument('--addr', default='90.90.176.152', type=str,
                        help='master addr')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='hccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to'
                             'launch N processes per node, which has N NPUs.'
                             'This is the fastest way to use PyTorch for'
                             'either single node or multi node data parallel'
                             'training')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--device_num', default=-1, type=int,
                        help='device num')
    args = parser.parse_args()
 

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29501'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device_list != '':
        npus_per_node = len(process_device_map)
    elif args.device_num > 0:
        npus_per_node = args.device_num
        print('npus_per_node: {}'.format(npus_per_node))
    else:
        npus_per_node = torch.npu.device_count()
        print('npus_per_node: {}'.format(npus_per_node))

    if args.multiprocessing_distributed:
        # world_size means nums of all devices or nums of processes
        #args.world_size = npus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, args))
        

def main_worker(npu, npus_per_node, args):
    args.sync_bn = True

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]
    args.batch_size = 32
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
   
    # multi-p code
    process_device_map = device_id_to_process_device_map(args.device_list)
    args.npu = process_device_map[npu]

    if npu is not None:
        print("[npu id:", npu, "]", "Use NPU: {} for training".format(npu))

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * npus_per_node + npu
    print("rank:", args.rank)
    dist.init_process_group(backend=args.dist_backend,
                            world_size=args.world_size, rank=args.rank)

    args.CALCULATE_DEVICE = 'npu:{}'.format(npu)
    print(args.CALCULATE_DEVICE)
    torch.npu.set_device(args.CALCULATE_DEVICE)

    print("args.batch_size = int(args.batch_size / npus_per_node) \n{} = int({}/{})".format(int(args.batch_size/npus_per_node),args.batch_size,npus_per_node))
    args.batch_size = int(args.batch_size / npus_per_node)
    
    args.workers = int((args.workers + npus_per_node - 1) / npus_per_node)
  
    
    trainer = Trainer(args)
    trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model,
                                                      device_ids=[args.npu],    # was args.npu
                                                      broadcast_buffers=False,find_unused_parameters=True)
    #trainer.model.to(torch.device(CALCULATE_DEVICE))
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.model = trainer.model.to(torch.device(args.CALCULATE_DEVICE))
        tbar = tqdm(trainer.train_loader)
        train_loss = 0.0
        num_img_tr = len(trainer.train_loader)
        pre_time = 0
        core_time = 0
        itr = 0
        for i, sample in enumerate(tbar):
          t_int = time.time()
          image, target = sample['image'], sample['label']
          image, target = image.to(args.CALCULATE_DEVICE, non_blocking=True), target.to(args.CALCULATE_DEVICE, non_blocking=True)
          
          trainer.scheduler(trainer.optimizer, i, epoch, trainer.best_pred)
          trainer.optimizer.zero_grad()
          output = trainer.model(image)
          output = output.permute(0, 2, 3, 1).reshape(-1,21)
          target = target.flatten().int()
          loss = trainer.criterion(output, target)
          loss.backward()
          trainer.optimizer.step()
          train_loss += loss.item()
          tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
          print('Train loss: %.3f' % (train_loss / (i + 1)))
          trainer.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
          trainer.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
          print('[Epoch: %d, numImages: %5d]' % (epoch, i * trainer.args.batch_size + image.data.shape[0]))
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch,args)

    trainer.writer.close()

if __name__ == "__main__":
   main()

