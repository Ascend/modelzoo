# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train CTPN and get checkpoint files."""
import argparse
import ast
import glob
import os
import shutil
import time

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, load_param_into_net, export
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, \
    ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed
from src.ctpn import CTPN, CTPN_Infer
from src.config import config, pretrain_config, finetune_config
from src.dataset import create_ctpn_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import LossCallBack, LossNet, WithLossCell, \
    TrainOneStepCell

import numpy as np
import moxing as mox

set_seed(1)

parser = argparse.ArgumentParser(description="CTPN training")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                    help="Run distribute, default: false.")
parser.add_argument("--pre_trained", type=str, default="",
                    help="Pretrained file path.")
parser.add_argument("--device_id", type=int, default=0,
                    help="Device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1,
                    help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0,
                    help="Rank id, default: 0.")
parser.add_argument("--task_type", type=str, default="Finetune",
                    choices=['Pretraining', 'Finetune'],
                    help="task type, default:Finetune")
parser.add_argument("--train_url", type=str, default="",
                    help="the path model saved on modelarts.")
parser.add_argument("--data_url", type=str, default="",
                    help="the training data directory on modelarts.")
parser.add_argument("--output_dir", type=str, default="/cache/ckpt",
                    help="model output path.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                    device_id=args_opt.device_id, save_graphs=True)

PRE_TRAIN_FILE = '/cache/MindRecord_CTPN_PRETRAIN/ctpn_pretrain.mindrecord0'
FINE_TUNE_FILE = '/cache/MindRecord_CTPN_FINETUNE/ctpn_finetune.mindrecord0'
TOTAL_EPOCH = 1
CKPT_OUTPUT_PATH = args_opt.output_dir
CKPT_OUTPUT_FILE_PATH = os.path.join(CKPT_OUTPUT_PATH, 'ckpt_0')


def set_param():
    if args_opt.run_distribute:
        rank_id = args_opt.rank_id
        dev_num = args_opt.device_num
        context.set_auto_parallel_context(
            device_num=dev_num, parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True)
        init()
    else:
        rank_id = 0
        dev_num = 1
    if args_opt.task_type == "Pretraining":
        print("Start to do pretraining")
        mind_rec_file = PRE_TRAIN_FILE
        train_cfg = pretrain_config
    else:
        print("Start to do finetune")
        mind_rec_file = FINE_TUNE_FILE
        train_cfg = finetune_config
    return rank_id, dev_num, mind_rec_file, train_cfg


def set_net_param(rank_id, dev_num, mind_rec_file, train_cfg):
    # When create MindDataset, using the first mindrecord file,
    # such as ctpn_pretrain.mindrecord0.
    data_set = create_ctpn_dataset(mind_rec_file, repeat_num=1,
                                   batch_size=config.batch_size,
                                   device_num=dev_num, rank_id=rank_id)
    data_set_size = data_set.get_dataset_size()
    train_net = CTPN(config=config, is_training=True)
    train_net = train_net.set_train()

    load_path = args_opt.pre_trained
    if args_opt.task_type == "Pretraining":
        print("load backbone vgg16 ckpt {}".format(args_opt.pre_trained))
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if not item.startswith('vgg16_feature_extractor'):
                param_dict.pop(item)
        load_param_into_net(train_net, param_dict)
    else:
        if load_path != "":
            print("load pretrain ckpt {}".format(args_opt.pre_trained))
            param_dict = load_checkpoint(load_path)
            load_param_into_net(train_net, param_dict)
    loss = LossNet()
    lr = Tensor(dynamic_lr(train_cfg, data_set_size), mstype.float32)
    opt = Momentum(params=train_net.trainable_params(), learning_rate=lr,
                   momentum=config.momentum, weight_decay=config.weight_decay,
                   loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(train_net, loss)
    if args_opt.run_distribute:
        train_net = TrainOneStepCell(net_with_loss, opt,
                                     sens=config.loss_scale, reduce_flag=True,
                                     mean=True, degree=dev_num)
    else:
        train_net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)
    return data_set_size, train_net, data_set


def model_trans():
    match_rule = "*.ckpt"
    placed_match_path = os.path.join(CKPT_OUTPUT_FILE_PATH, match_rule)
    placed_ckpt_list = glob.glob(placed_match_path)
    if not placed_ckpt_list:
        print("ckpt file not exist.")
        return
    placed_ckpt_list.sort(key=lambda fn: os.path.getmtime(fn))
    ckpt_path = placed_ckpt_list[-1]
    print("ckpt path is %s" % ckpt_path)

    ctpn_net = CTPN_Infer(config=config)
    param_dict = load_checkpoint(ckpt_path)
    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(ctpn_net, param_dict_new)
    img = Tensor(np.zeros(
        [config.test_batch_size, 3, config.img_height, config.img_width]),
                 ms.float16)
    export(ctpn_net, img, file_name='ctpn', file_format='AIR')
    shutil.copy('ctpn.air', CKPT_OUTPUT_PATH)


if __name__ == '__main__':
    mox.file.copy_parallel(args_opt.data_url, '/cache')

    rank, device_num, mind_record_file, training_cfg = set_param()
    print("CHECKING MINDRECORD FILES ...")
    while not os.path.exists(mind_record_file + ".db"):
        time.sleep(5)
    print("CHECKING MINDRECORD FILES DONE!")

    dataset_size, net, dataset = set_net_param(rank, device_num,
                                               mind_record_file, training_cfg)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(args_opt.output_dir,
                                            "ckpt_" + str(rank) + "/")
        ckpt_cb = ModelCheckpoint(
            prefix='ctpn', directory=save_checkpoint_path, config=ckpt_config)
        cb += [ckpt_cb]

    model = Model(net)
    model.train(TOTAL_EPOCH, dataset, callbacks=cb, dataset_sink_mode=True)
    model_trans()

    if not os.path.exists(CKPT_OUTPUT_PATH):
        os.makedirs(CKPT_OUTPUT_PATH, 0o755)
    mox.file.copy_parallel(CKPT_OUTPUT_PATH, args_opt.train_url)
