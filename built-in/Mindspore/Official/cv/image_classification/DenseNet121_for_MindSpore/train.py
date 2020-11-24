# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train launch."""
import os
import time
import argparse
import datetime

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import set_seed

from src.optimizers import get_param_groups
from src.network import DenseNet121
from src.datasets import classification_dataset
from src.losses.crossentropy import CrossEntropy
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.utils.logging import get_logger
from src.config import config

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target="Davinci", save_graphs=False, device_id=devid)

set_seed(1)

class BuildTrainNetwork(nn.Cell):
    """build training network"""
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss

class ProgressMonitor(Callback):
    """monitor loss and time"""
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
        """process epoch end"""
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = self.args.per_batch_size * (me_step-self.me_epoch_start_step_num) * self.args.group_size / time_used
        self.args.logger.info('epoch[{}], iter[{}], loss:{},'
                              'mean_fps:{:.2f} imgs/sec'.format(real_epoch, me_step, cb_params.net_outputs, fps_mean))
        if self.args.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.args.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info('epoch[{}], iter[{}], loss:{}, ckpt:{},'
                                      'ckpt_fn:{}'.format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn))

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info('end network train...')


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore classification training')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='train data dir')

    # network related
    parser.add_argument('--pretrained', default='', type=str, help='model_path, local pretrained model to load')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')

    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)
    args.image_size = config.image_size
    args.num_classes = config.num_classes
    args.lr = config.lr
    args.lr_scheduler = config.lr_scheduler
    args.lr_epochs = config.lr_epochs
    args.lr_gamma = config.lr_gamma
    args.eta_min = config.eta_min
    args.T_max = config.T_max
    args.max_epoch = config.max_epoch
    args.warmup_epochs = config.warmup_epochs
    args.weight_decay = config.weight_decay
    args.momentum = config.momentum
    args.is_dynamic_loss_scale = config.is_dynamic_loss_scale
    args.loss_scale = config.loss_scale
    args.label_smooth = config.label_smooth
    args.label_smooth_factor = config.label_smooth_factor
    args.ckpt_interval = config.ckpt_interval
    args.ckpt_path = config.ckpt_path
    args.is_save_on_master = config.is_save_on_master
    args.rank = config.rank
    args.group_size = config.group_size
    args.log_interval = config.log_interval
    args.per_batch_size = config.per_batch_size

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.image_size = list(map(int, args.image_size.split(',')))

    return args

def merge_args(args, cloud_args):
    """dictionary"""
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

def train(cloud_args=None):
    """training process"""
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
    de_dataset = classification_dataset(args.data_dir, args.image_size,
                                        args.per_batch_size, args.max_epoch,
                                        args.rank, args.group_size)
    de_dataset.map_model = 4
    args.steps_per_epoch = de_dataset.get_dataset_size()

    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')
    # get network and init
    network = DenseNet121(args.num_classes)
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
        lr_scheduler = MultiStepLR(args.lr,
                                   args.lr_epochs,
                                   args.lr_gamma,
                                   args.steps_per_epoch,
                                   args.max_epoch,
                                   warmup_epochs=args.warmup_epochs)
    elif args.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(args.lr,
                                         args.T_max,
                                         args.steps_per_epoch,
                                         args.max_epoch,
                                         warmup_epochs=args.warmup_epochs,
                                         eta_min=args.eta_min)
    else:
        raise NotImplementedError(args.lr_scheduler)
    lr_schedule = lr_scheduler.get_lr()

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr_schedule),
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

    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size,
                                      gradients_mean=True)
    model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O3")

    # checkpoint save
    progress_cb = ProgressMonitor(args)
    callbacks = [progress_cb,]
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
