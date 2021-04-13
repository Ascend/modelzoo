# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

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
import argparse
import moxing as mox
import os

import tensorflow as tf
import glob
import numpy as np

import sys
import shutil
import ast

import alexnet.data_loader as dl
import alexnet.model as ml
import alexnet.hyper_param as hp
import alexnet.layers as ly
import alexnet.logger as lg
import alexnet.trainer as tr
import alexnet.create_session as cs


CUR_PATH = os.path.dirname(os.path.realpath(__file__))
CKPT_OUTPUT_PATH = "/cache/ckpt_alexnet"
PB_OUTPUT_PATH = "/cache/pb_alexnet"


def set_env():
    os.environ['DEVICE_INDEX'] = os.getenv('RANK_ID')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # 数据集目录
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')

    parser.add_argument('--iterations_per_loop', default=100, type=int,
                        help="""the number of steps in devices for each iteration""")
    parser.add_argument('--rank_size', default=1, type=int,
                        help="""number of NPUs  to use.""")
    parser.add_argument('--shard', default=False, type=ast.literal_eval,
                        help="""whether to use shard or not""")
    parser.add_argument('--mode', default='train',
                        help="""mode to run the program  e.g. train, evaluate, and 
                        train_and_evaluate """)
    parser.add_argument('--epochs_between_evals', default=5, type=int,
                        help="""the interval between train and evaluation , only meaningful 
                        when the mode is train_and_evaluate """)
    parser.add_argument('--data_dir', default='/cache/data_dir_alexnet',
                        help="""directory to data.""")
    parser.add_argument('--dtype', default=tf.float32,
                        help="""data type of inputs.""")
    parser.add_argument('--use_nesterov', default=True, type=ast.literal_eval,
                        help=""" used in optimizer""")
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help="""label smoothing factor""")
    parser.add_argument('--weight_decay', default=0.0001,
                        help="""weight decay""")
    parser.add_argument('--batch_size', default=256, type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--max_train_steps', default=None, type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help="""number of warmup epochs """)
    parser.add_argument('--lr', default=0.015, type=float,
                        help="""learning rate""")
    parser.add_argument('--max_epochs', default=150, type=int,
                        help="""total epochs for training""")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="""momentum used in optimizer.""")

    parser.add_argument('--synthetic', default=False, type=ast.literal_eval,
                        help="""parser file used.""")

    parser.add_argument('--debug', default=True, type=ast.literal_eval,
                        help="""parser file used.""")
    parser.add_argument('--eval', default=False, type=ast.literal_eval,
                        help="""parser file used.""")

    parser.add_argument('--display_every', default=1, type=int,
                        help="""the frequency to display info""")
    parser.add_argument('--log_name', default='alexnet_training.log',
                        help="""name of log file""")
    parser.add_argument('--alexnet_version', default='he_uniform',
                        help="""version of weight initialization""")

    parser.add_argument('--log_dir', default=CKPT_OUTPUT_PATH,
                        help="""log directory""")
    parser.add_argument('--save_summary_steps', default=100, type=int,
                        help="""frequency to save summary""")
    parser.add_argument('--max_checkpoint_to_save', default=5, type=int,
                        help="""frequency to save checkpoints""")
    parser.add_argument('--checkpoint_dir', default=CKPT_OUTPUT_PATH,
                        help="""directory to checkpoints""")

    parser.add_argument('--num_classes', default=1000, type=int,
                        help="""number of classes for datasets """)
    parser.add_argument('--save_checkpoints_steps', default=1000, type=int,
                        help="""frequency to save checkpoints""")
    parser.add_argument('--display', default=1, type=int,
                        help="""the frequency to display info""")
    parser.add_argument('--do_checkpoint', default=True, type=ast.literal_eval,
                        help="""whether to save checkpoints or not.""")
    parser.add_argument('--freeze_pb', default=True, type=ast.literal_eval,
                        help="""whether to freeze checkpoints to pb model""")
    # 配置预训练ckpt路径
    parser.add_argument('--restore_path', default='',
                        help="""restore path of pretrained model""")
    # 不加载预训练网络中FC层权重
    parser.add_argument('--restore_exclude', default=['dense_2'],
                        help="""restore_exclude""")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def find_latest_ckpt(ckpt_dir=CKPT_OUTPUT_PATH):
    match_rule = "model.ckpt-[!0]*.meta*"
    ckpt_match_path = os.path.join(ckpt_dir, match_rule)
    ckpt_list = glob.glob(ckpt_match_path)
    if not ckpt_list:
        print("ckpt file not generated.")
        return
    ckpt_list.sort(key=lambda x: os.path.getmtime(x))
    return ckpt_list[-1].rsplit('.', 1)[0]


def frozen_ckpt_to_pb(args):
    print("start to freeze ckpt to pb")
    frozen_py_file = os.path.join(CUR_PATH, "frozen_graph.py")
    ckpt_path = "--ckpt_path=" + find_latest_ckpt()
    output_pb_path = "--output_path=" + PB_OUTPUT_PATH
    num_classes = "--num_classes=" + str(args.num_classes)

    if not os.path.exists(PB_OUTPUT_PATH):
        os.makedirs(PB_OUTPUT_PATH, exist_ok=True)

    # 将ckpt转换为pb模型
    frozen_cmd = " ".join(["python3.7", frozen_py_file, ckpt_path, output_pb_path, num_classes])
    print('frozen_cmd: {}'.format(frozen_cmd))
    os.system(frozen_cmd)

    res_file_list = os.listdir(PB_OUTPUT_PATH)
    if not res_file_list or 'alexnet_tf_910.pb' not in res_file_list:
        print("freeze ckpt to pb failed: {}".format(res_file_list))
        return

    # 拷贝pb模型到模型输出目录
    mox.file.copy_parallel(PB_OUTPUT_PATH, args.train_url)
    print("freeze ckpt to pb successfully")


def main():
    # 设置环境变量
    set_env()

    args = parse_args()
    args.global_batch_size = args.batch_size * args.rank_size

    # 现将数据集拷贝到ModelArts指定读取的cache目录
    mox.file.copy_parallel(args.data_url, '/cache/data_dir_alexnet')
    args.data_dir = '/cache/data_dir_alexnet'
    # modelarts平台上只能指定restore_path的ckpt模型名字，所以需要拼接完整路径
    if args.restore_path:
        args.restore_path = os.path.join(args.data_dir, 'ckpt_pretrained', args.restore_path)
        match_rule = args.restore_path + "*"
        if not glob.glob(match_rule):
            print("restore_path: {} not exists".format(match_rule))
            return

    # 设置日志打印级别
    tf.logging.set_verbosity(tf.logging.INFO)

    session = cs.CreateSession(args)
    data = dl.DataLoader(args)
    hyper_param = hp.HyperParams(args)
    layers = ly.Layers()
    logger = lg.LogSessionRunHook(args)
    model = ml.Model(args, data, hyper_param, layers, logger)
    trainer = tr.Trainer(session, args, data, model, logger)

    if args.mode == 'train':
        trainer.train()
        # 训练完成后把生成的模型拷贝到指导输出目录
        if not os.path.exists(os.path.dirname(CKPT_OUTPUT_PATH)):
            os.makedirs(os.path.dirname(CKPT_OUTPUT_PATH), exist_ok=True)
        mox.file.copy_parallel(CKPT_OUTPUT_PATH, args.train_url)
        if args.freeze_pb:
            frozen_ckpt_to_pb(args)
    elif args.mode == 'evaluate':
        trainer.evaluate()
    else:
        raise ValueError('Invalid type of mode')


if __name__ == '__main__':
    main()
