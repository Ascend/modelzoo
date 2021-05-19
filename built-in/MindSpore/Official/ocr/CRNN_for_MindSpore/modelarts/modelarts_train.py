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
"""crnn training"""
import argparse
import glob
import os
import shutil

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, export
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import TimeMonitor, LossMonitor,\
    CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_group_size, get_rank

import numpy as np
import moxing as mox

from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.crnn_for_train import TrainOneStepCellWithGradClip

set_seed(1)

parser = argparse.ArgumentParser(description="crnn training")
parser.add_argument('--train_url', type=str, default='',
                    help='the path model saved')
parser.add_argument('--data_url', type=str, default='',
                    help='the training data')
parser.add_argument("--run_distribute", action='store_true',
                    help="Run distribute, default is false.")
parser.add_argument('--dataset_path', type=str, default='/cache',
                    help='Dataset path, default is None')
parser.add_argument('--platform', type=str, default='Ascend',
                    choices=['Ascend'],
                    help='Running platform, only support Ascend now. '
                         'Default is Ascend.')
parser.add_argument('--model', type=str, default='lowercase',
                    help="Model type, default is lowercase")
parser.add_argument('--dataset', type=str, default='ic13',
                    choices=['synth', 'ic03', 'ic13', 'svt', 'iiit5k'])
parser.add_argument("--pre_trained", type=str, default="",
                    help="Pretrain file path.")

parser.add_argument("--output_dir", type=str, default="/cache/ckpt",
                    help="model output path.")
parser.add_argument("--max_text_length", type=int, default=23,
                    help="max number of digits in each.")
parser.add_argument("--image_width", type=int, default=100,
                    help="width of text images.")
parser.add_argument("--image_height", type=int, default=32,
                    help="height of text images.")
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size of input tensor.")
parser.add_argument("--epoch_size", type=int, default=10,
                    help="only valid for taining, which is always 1.")
parser.add_argument("--hidden_size", type=int, default=256,
                    help="hidden size in LSTM layers.")
parser.add_argument("--learning_rate", type=float, default=0.02,
                    help="momentum of SGD optimizer.")
parser.add_argument("--momentum", type=float, default=0.95,
                    help="learning rate.")
parser.set_defaults(run_distribute=False)
args_opt = parser.parse_args()

if args_opt.model == 'lowercase':
    from src.config import config1 as config
else:
    from src.config import config2 as config
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform,
                    save_graphs=False)
if args_opt.platform == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

CKPT_OUTPUT_PATH = args_opt.output_dir
CKPT_OUTPUT_FILE_PATH = os.path.join(CKPT_OUTPUT_PATH, 'ckpt_0')


def set_config(config):
    # Sets the passed super parameter
    config.max_text_length = args_opt.max_text_length
    config.image_width = args_opt.image_width
    config.image_height = args_opt.image_height
    config.batch_size = args_opt.batch_size
    config.epoch_size = args_opt.epoch_size
    config.hidden_size = args_opt.hidden_size
    config.learning_rate = args_opt.learning_rate
    config.momentum = args_opt.momentum

    print("config.max_text_length is %d" % config.max_text_length)
    print("config.image_width is %d" % config.image_width)
    print("config.image_height is %d" % config.image_height)
    print("config.batch_size is %d" % config.batch_size)
    print("config.epoch_size is %d" % config.epoch_size)
    print("config.hidden_size is %d" % config.hidden_size)
    print("config.learning_rate is %f" % config.learning_rate)
    print("config.momentum is %f" % config.momentum)
    return


def set_device():
    if args_opt.run_distribute:
        if args_opt.platform == 'Ascend':
            init()
            device_num = int(os.environ.get("RANK_SIZE"))
            rank = int(os.environ.get("RANK_ID"))
        else:
            init()
            device_num = get_group_size()
            rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True)
    else:
        device_num = 1
        rank = 0
    return device_num, rank


def set_net_param(device_num, rank):
    # create dataset
    dataset = create_dataset(name=args_opt.dataset,
                             dataset_path=args_opt.dataset_path,
                             batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank,
                             config=config)
    step_size = dataset.get_dataset_size()
    # define lr
    lr_init = config.learning_rate
    lr = nn.dynamic_lr.cosine_decay_lr(0.0, lr_init,
                                       config.epoch_size * step_size, step_size,
                                       config.epoch_size)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=config.max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr,
                 momentum=config.momentum, nesterov=config.nesterov)

    load_path = args_opt.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if not (item.startswith('backbone') or item.startswith(
                        'crnn_mask')):
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)

    net = WithLossCell(net, loss)
    net = TrainOneStepCellWithGradClip(net, opt).set_train()
    return net, step_size, dataset


def main():
    device_num, rank = set_device()

    # Sets the passed super parameter
    set_config(config)

    # Sets the network parameter
    net, step_size, dataset = set_net_param(device_num, rank)

    # define model
    model = Model(net)
    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if config.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_steps,
            keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(args_opt.output_dir,
                                      'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(prefix="crnn", directory=save_ckpt_path,
                                  config=config_ck)
        callbacks.append(ckpt_cb)
    model.train(config.epoch_size, dataset, callbacks=callbacks)


def model_trans():
    paths = os.listdir(CKPT_OUTPUT_PATH)
    print("output path files is %s" % paths)

    match_rule = "*.ckpt"
    placed_match_path = os.path.join(CKPT_OUTPUT_FILE_PATH, match_rule)
    print("placed_match_path is %s" % placed_match_path)
    placed_ckpt_list = glob.glob(placed_match_path)
    print("placed_ckpt_list is %s" % placed_ckpt_list)
    if not placed_ckpt_list:
        print("ckpt file not exist.")
        return
    placed_ckpt_list.sort(key=os.path.getmtime)
    ckpt_path = placed_ckpt_list[-1]
    print("ckpt path is %s" % ckpt_path)

    config.batch_size = 1
    net = crnn(config)

    load_checkpoint(ckpt_path, net=net)
    net.set_train(False)

    input_data = Tensor(
        np.zeros([1, 3, config.image_height, config.image_width]), ms.float32)

    export(net, input_data, file_name='crnn', file_format='AIR')
    shutil.copy('crnn.air', CKPT_OUTPUT_PATH)


if __name__ == '__main__':
    # copy dataset to the /cache dir
    mox.file.copy_parallel(args_opt.data_url, '/cache')
    main()
    model_trans()
    # after training, copy model to the output dir
    if not os.path.exists(CKPT_OUTPUT_PATH):
        os.makedirs(CKPT_OUTPUT_PATH, exist_ok=True)
    mox.file.copy_parallel(CKPT_OUTPUT_PATH, args_opt.train_url)
