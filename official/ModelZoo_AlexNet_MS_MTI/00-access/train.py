import os
from mindspore import context
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, 
                    device_target="Davinci", save_graphs=True, device_id=devid)
loglevel = 'error'
os.system('su root -c "adc --host 127.0.0.1:22118 --log \\"SetLogLevel(0)[{}]\\""'.format(loglevel))
os.system('su root -c "adc --host 127.0.0.1:22118 --log \\"SetLogLevel(0)[{}]\\" --device 4"'.format(loglevel))

import time
import argparse
import datetime

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.callback import Callback
try:
    from mindspore.train._loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
except:
    from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager

from alexnet.datasets import classification_dataset_c
from alexnet.losses.crossentropy import CrossEntropy
from alexnet.lr_scheduler.warmup_step_lr import warmup_step_lr
from alexnet.lr_scheduler.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
from alexnet.utils.logging import get_logger
from alexnet.optimizers import get_param_groups
from alexnet.network.alexnet import get_network
from alexnet.utils.mixed_precision import mixed_precision_warpper

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion
 
    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss

class ProgressMonitor(Callback):
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args
        self.ckpt_history = []

    def begin(self, run_context):
        self.args.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.args.per_batch_size * (me_step-self.me_epoch_start_step_num) * self.args.group_size / time_used
        self.args.logger.info('epoch[{}], maiter[{}], loss:{}, mean_fps:{:.2f} imgs/sec'.format(real_epoch, me_step, cb_params.net_outputs, fps_mean))

        if self.args.rank_save_ckpt_flag:
            try:
                import moxing as mox
                import glob
                ckpts = glob.glob(os.path.join(self.args.outputs_dir, '*.ckpt'))
                for ckpt in ckpts:
                    ckpt_fn = os.path.basename(ckpt)
                    if not ckpt_fn.startswith('{}-'.format(self.args.rank)):
                        continue
                    if ckpt in self.ckpt_history:
                        continue
                    self.ckpt_history.append(ckpt)
                    self.args.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{}, ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))
            except:
                self.args.logger.info('local passed')

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        # self.args.logger.info('start step...')
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info('end network train...')


def parse_args(cloud_args={}):
    parser = argparse.ArgumentParser('mindspore classification training')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='train data dir')
    parser.add_argument('--num_classes', type=int, default=1000, help='num of classes in dataset')
    parser.add_argument('--image_size', type=str, default='224,224', help='image size of the dataset')
    parser.add_argument('--per_batch_size', default=256, type=int, help='batch size for per gpu')

    # network related
    parser.add_argument('--backbone', default='alexnet', type=str, help='backbone')
    parser.add_argument('--pretrained', default='', type=str, help='model_path, local pretrained model to load')

    # optimizer and lr related
    parser.add_argument('--lr_scheduler', default='cosine_annealing', type=str,
                        help='lr-scheduler, option type: exponential, cosine_annealing')
    parser.add_argument('--lr', default=0.13, type=float, help='learning rate of the training')
    parser.add_argument('--lr_epochs', type=str, default='30,60,90,120', help='epoch of lr changing')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='decrease lr by a factor of exponential lr_scheduler')
    parser.add_argument('--eta_min', type=float, default=0., help='eta_min in cosine_annealing scheduler')
    parser.add_argument('--T_max', type=int, default=150, help='T-max in cosine_annealing scheduler')
    parser.add_argument('--max_epoch', type=int, default=150, help='max epoch num to train the model')
    parser.add_argument('--warmup_epochs', default=5, type=float, help='warmup epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # loss related
    parser.add_argument('--is_dynamic_loss_scale', type=int, default=0, help='dynamic loss scale')
    parser.add_argument('--loss_scale', type=int, default=1024, help='static loss scale')
    parser.add_argument('--label_smooth', type=int, default=1, help='whether to use label smooth in CE')
    parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='smooth strength of original one-hot')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=1000, help='ckpt_interval')
    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.image_size = list(map(int, args.image_size.split(',')))

    return args


def merge_args(args, cloud_args):
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args
	

def train(cloud_args={}):
    args = parse_args(cloud_args)

    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

    if args.is_dynamic_loss_scale == 1:
        args.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt
    
    # select for master rank save ckpt or all rank save, compatiable for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path, 
        datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)

    # dataloader
    de_dataset = classification_dataset_c(args.data_dir, args.image_size, 
                                  args.per_batch_size, args.max_epoch,
                                  args.rank, args.group_size)
    de_dataset.map_model = 4  # !!!important
    args.steps_per_epoch = de_dataset.get_dataset_size()

    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')
    # get network and init
    network = get_network(args.backbone, args.num_classes)
    # loss
    if not args.label_smooth:
        args.label_smooth_factor = 0.0
    criterion = CrossEntropy(smooth_factor=args.label_smooth_factor,
                             num_classes=args.num_classes)

    # load pretrain model
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load model {} success'.format(args.pretrained))

    # lr scheduler
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.T_max,
                                        args.eta_min)
    else:
        raise NotImplementedError(args.lr_scheduler)

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr), 
                   momentum=args.momentum, 
                   weight_decay=args.weight_decay, 
                   loss_scale=args.loss_scale)

    # mixed precision training 
    criterion.add_flags_recursive(fp32=True)

    # package training process, adjust lr + forward + backward + optimizer
    train_net = BuildTrainNetwork(network, criterion)
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE
    if args.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
		
    # Model api changed since TR5_branch 2020/03/09
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size, parameter_broadcast=True, mirror_mean=True)
    model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager)

    # checkpoint save
    progress_cb = ProgressMonitor(args)
    callbacks = [progress_cb, ]
    if args.rank_save_ckpt_flag:
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval,
                                        keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                    directory=args.outputs_dir,
                                    prefix='{}'.format(args.rank))
        callbacks.append(ckpt_cb)

    model.train(args.max_epoch, de_dataset, callbacks=callbacks)


if __name__ == "__main__":
    train()
