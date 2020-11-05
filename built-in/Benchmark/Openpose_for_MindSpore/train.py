# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from mindspore import context
import time
import argparse
import datetime
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.dataset import openpose 
from src.openposenet import OpenPoseNet
from src.loss import openpose_loss,TrainingWrapper,LossCallBack,BuildTrainNetwork
from src.config import params
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target="Ascend", save_graphs=False, device_id=devid)

def parse_args():
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore openpose training')

    # dataset related
    parser.add_argument('--train_dir', type=str, help='train data dir')
    parser.add_argument('--train_ann', type=str, help='train annotations json')
    # profiler init
    parser.add_argument('--need_profiler', type=int, default=0, help='whether use profiler')
    # logging related
    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args, _ = parser.parse_known_args()
    args.lr_steps = list(map(int, params["lr_steps"].split(',')))
    
    args.jsonpath_train = os.path.join(params['data_dir'], 'annotations/'+ args.train_ann)
    args.imgpath_train = os.path.join(params['data_dir'], args.train_dir)
    args.maskpath_train = os.path.join(params['data_dir'], 'ignore_mask_train')

    return args
    
def get_lr(lr,lr_gamma,steps_per_epoch,max_epoch_train,lr_steps,group_size):
    lr_stage = np.array([lr]*steps_per_epoch*max_epoch_train).astype('f')
    for step in lr_steps:
        step //= group_size
        lr_stage[step:] *= lr_gamma

    lr_base = lr_stage.copy()
    lr_base = lr_base/4
    
    lr_vgg = lr_base.copy()
    vgg_freeze_step = 2000 // group_size
    lr_vgg[:vgg_freeze_step] = 0
    print("len_of_lr",len(lr_stage),len(lr_base),len(lr_vgg))
    return lr_stage,lr_base,lr_vgg
    
def load_model(test_net,model_path):
    if model_path:
        param_dict = load_checkpoint(model_path)
        #print(type(param_dict))
        param_dict_new = {}
        for key, values in param_dict.items():
            #print('key:', key)
            if key.startswith('moment'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
                
            # else:
                # param_dict_new[key] = values
        load_param_into_net(test_net,param_dict_new)

class show_loss_list():
    def __init__(self,name):
        self.loss_list = np.zeros(6).astype('f')
        self.sums = 0
        self.name = name
            
    def add(self,list_of_tensor):
        self.sums += 1
        for i, loss_tensor in enumerate(list_of_tensor):
            self.loss_list[i] += loss_tensor.asnumpy()
    def show(self):
        print(self.name + ' stage_loss:',self.loss_list/(self.sums+1e-8),flush=True)
        self.loss_list = np.zeros(6).astype('f')
        self.sums = 0
class AverageMeter():
    def __init__(self):
        self.loss = 0
        self.sum = 0
    def add(self,tensor):
        self.sum += 1
        self.loss += tensor.asnumpy()
    def meter(self):
        avergeLoss = self.loss/(self.sum+1e-8)
        self.loss = 0
        self.sum = 0
        return avergeLoss
        
def train():
    """Train function."""
    args = parse_args()
    
    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        
    # select for master rank save ckpt or all rank save, compatiable for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1
     
    ## logger
    args.outputs_dir = os.path.join(params['save_model_path'], datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    
    if args.need_profiler:
        from mindspore.profiler import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)

    
    # create network
    print('start create network')
    criterion = openpose_loss()
    criterion.add_flags_recursive(fp32=True)
    network = OpenPoseNet(vggpath = params['vgg_path'])
    #network.add_flags_recursive(fp16=True)
    if params["load_pretrain"]:
        print("load pretrain model:",params["pretrained_model_path"])
        load_model(network, params["pretrained_model_path"])
    train_net = BuildTrainNetwork(network, criterion)
    
    # create dataset
    if os.path.exists(args.jsonpath_train) and os.path.exists(args.imgpath_train) and os.path.exists(args.maskpath_train):
        print('start create dataset')
    else:
        print('wrong data path')
        return 0
    de_dataset_train,steps_per_epoch = openpose(args.jsonpath_train, args.imgpath_train, args.maskpath_train, per_batch_size=params['batch_size'], max_epoch=params["max_epoch_train"],rank = args.rank,group_size = args.group_size)
    #de_dataset_train = openpose.mindrecord( per_batch_size=params['batch_size'], max_epoch=params["max_epoch_train"])
    print("step",steps_per_epoch)
    de_dataloader_train = de_dataset_train.create_tuple_iterator()
    
    # lr scheduler
    lr_stage,lr_base,lr_vgg =get_lr(params['lr'],params['lr_gamma'],steps_per_epoch,params["max_epoch_train"],args.lr_steps,args.group_size)
    vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.trainable_params()))
    base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.trainable_params()))
    stages_params = list(filter(lambda x: 'base' not in x.name, train_net.trainable_params()))

    group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
                    {'params': base_params, 'lr': lr_base},
                    {'params': stages_params, 'lr': lr_stage}]
    
    opt = Adam(group_params,loss_scale=params['loss_scale'])
    
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    else:
        parallel_mode = ParallelMode.STAND_ALONE
        degree = 1
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
    train_net = TrainingWrapper(train_net, opt, sens=params['loss_scale'])
    
    if args.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = params["keep_checkpoint_max"]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=params['ckpt_interval'],
                                       keep_checkpoint_max= ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=args.outputs_dir,
                                  prefix='{}'.format(args.rank))
        cb_params = _InternalCallbackParam()
        cb_params.train_network = train_net
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)
        
    train_net.set_train()
    #op = ParameterReduce()
    loss_meter = AverageMeter()
    
    old_progress = -1
    t_end = time.time()
    time_end = time.time()
    show_heatmaps_loss = show_loss_list("htmp")
    show_pafs_loss = show_loss_list("pafs")
    print('start training, first step maybe take hundreds of seconds, please wait')
    for i, (img, pafs, heatmaps, mask) in enumerate(de_dataloader_train):
     
        loss, heatmaps_loss, pafs_loss = train_net(img, pafs, heatmaps,mask)
        
        #add to loss meter
        loss_meter.add(loss)
        show_heatmaps_loss.add(heatmaps_loss)
        show_pafs_loss.add(pafs_loss)
        if args.rank_save_ckpt_flag:
            # ckpt progress
            cb_params.cur_step_num = (i + 1)*args.group_size  # current step number
            cb_params.batch_num = (i + 2)*args.group_size
            ckpt_cb.step_end(run_context)
            
        if i % params['log_interval'] == 0:
            time_used = time.time()- time_end
            epoch = int(i / steps_per_epoch)
            fps = params['batch_size'] * (i - old_progress) * args.group_size / time_used
            print('epoch[{}], iter[{}], loss[{}], {:.2f} imgs/sec, vgglr={},baselr={},stagelr={}'.format(epoch, i*args.group_size, loss_meter.meter(),fps, lr_vgg[i],lr_base[i],lr_stage[i]),flush = True)
            show_pafs_loss.show()
            show_heatmaps_loss.show()
            
            old_progress = i
            time_end = time.time()
        if (i + 1) % steps_per_epoch == 0 and args.rank_save_ckpt_flag:
            cb_params.cur_epoch_num += 1

        if args.need_profiler:
            if i == 100:
                profiler.analyse()   
                break
    
if __name__ == "__main__":
    train()
