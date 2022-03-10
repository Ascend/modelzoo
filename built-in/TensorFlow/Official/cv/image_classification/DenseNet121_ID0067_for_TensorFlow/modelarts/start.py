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
import os

import tensorflow as tf

import glob
import ast

import densenet.data_loader as dl
import densenet.model as ml
import densenet.hyper_param as hp
import densenet.layers as ly
import densenet.logger as lg
import densenet.trainer as tr
import densenet.create_session as cs

import argparse
import moxing as mox


CUR_PATH = os.path.dirname(os.path.realpath(__file__))
CKPT_OUTPUT_PATH = "/cache/ckpt_densenet"
PB_OUTPUT_PATH = "/cache/pb_densenet"


def parse_args():
    """parse args from command line"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # 数据集目录
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    # 数据集分类数目
    parser.add_argument('--num_classes', type=int, default=1000,
                        help="""number of classes for datasets""")
    # 配置预训练ckpt路径
    parser.add_argument('--restore_path', type=str, default='',
                        help="""restore path""")
    # 不加载预训练网络中FC层权重
    parser.add_argument('--restore_exclude', default=['linear/', 'global_step'], help="""restore_exclude""")

    parser.add_argument('--rank_size', default=1, type=int,
                        help="""number of NPUs  to use.""")

    # mode and parameters related
    parser.add_argument('--mode', default='train',
                        help="""mode to run the program  e.g. train, evaluate, and
                        train_and_evaluate""")
    parser.add_argument('--max_train_steps', default=100, type=int,
                        help="""train steps for one NPU""")
    parser.add_argument('--iterations_per_loop', default=10, type=int,
                        help="""the number of steps in devices for each iteration""")
    parser.add_argument('--max_epochs', default=None, type=int,
                        help="""total epochs for training""")
    parser.add_argument('--epochs_between_evals', default=5, type=int,
                        help="""the interval between train and evaluation , only meaningful
                        when the mode is train_and_evaluate""")

    # dataset
    parser.add_argument('--data_dir', default='/cache/data_dir_densenet',
                        help="""directory to data.""")

    # path for evaluation
    parser.add_argument('--eval_dir', default='/cache/data_dir_densenet',
                        help="""directory to evaluate.""")

    parser.add_argument('--dtype', default=tf.float32,
                        help="""data type of inputs.""")
    parser.add_argument('--use_nesterov', default=True, type=ast.literal_eval,
                        help=""" used in optimizer""")
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help="""label smoothing factor""")
    parser.add_argument('--weight_decay', default=0.0001,
                        help="""weight decay""")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="""batch size for one NPU""")

    # learning rate and momentum
    parser.add_argument('--lr', default=0.1, type=float,
                        help="""learning rate""")
    parser.add_argument('--T_max', default=150, type=int,
                        help="""T_max for cosing_annealing learning rate""")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="""momentum used in optimizer.""")
    # display frequency
    parser.add_argument('--display_every', default=1, type=int,
                        help="""the frequency to display info""")
    # log file
    parser.add_argument('--log_name', default='densenet121_training.log',
                        help="""name of log file""")
    parser.add_argument('--log_dir', default='/cache/ckpt_densenet',
                        help="""log directory""")
    # 训练结束后是否根据最新的ckpt模型生成pb模型
    parser.add_argument('--freeze_pb', default=True, type=ast.literal_eval,
                        help="""whether to freeze checkpoints to pb model""")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def find_latest_ckpt(ckpt_dir=CKPT_OUTPUT_PATH):
    """find latest ckpt under ckpt_dir"""
    match_rule = "model.ckpt-[!0]*.meta*"
    ckpt_match_path = os.path.join(ckpt_dir, match_rule)
    ckpt_list = glob.glob(ckpt_match_path)
    if not ckpt_list:
        print("ckpt file not generated.")
        return
    ckpt_list.sort(key=os.path.getmtime)
    return ckpt_list[-1].rsplit('.', 1)[0]


def freeze_ckpt_to_pb(args):
    """freeze model from ckpt to pb"""
    print("start to freeze ckpt to pb")
    frozen_py_file = os.path.join(CUR_PATH, "densenet", "frozen_graph.py")
    ckpt_path = "--ckpt_path=" + find_latest_ckpt()
    output_pb_path = "--output_path=" + PB_OUTPUT_PATH
    num_classes = "--num_classes=" + str(args.num_classes)

    if not os.path.exists(PB_OUTPUT_PATH):
        os.makedirs(PB_OUTPUT_PATH, mode=0o755)

    # 将ckpt转换为pb模型
    frozen_cmd = " ".join(["python3.7", frozen_py_file, ckpt_path, output_pb_path, num_classes])
    print('frozen_cmd: {}'.format(frozen_cmd))
    os.system(frozen_cmd)

    res_file_list = os.listdir(PB_OUTPUT_PATH)
    if not res_file_list or 'densenet121_tf_910.pb' not in res_file_list:
        print("freeze ckpt to pb failed: {}".format(res_file_list))
        return

    # 拷贝pb模型到模型输出目录
    mox.file.copy_parallel(PB_OUTPUT_PATH, args.train_url)
    print("freeze ckpt to pb successfully")


def set_env():
    """set env variable"""
    os.environ['DEVICE_INDEX'] = os.getenv('RANK_ID')


def main():
    set_env()

    args = parse_args()
    args.global_batch_size = args.batch_size * args.rank_size

    # 现将数据集拷贝到ModelArts指定读取的cache目录
    mox.file.copy_parallel(args.data_url, '/cache/data_dir_densenet')
    args.data_dir = '/cache/data_dir_densenet'
    # modelarts平台上只能指定restore_path的ckpt模型名字，所以需要拼接完整路径
    if args.restore_path:
        args.restore_path = os.path.join(args.data_dir, 'ckpt_pretrained', args.restore_path)
        match_rule = args.restore_path + "*"
        if not glob.glob(match_rule):
            print("restore_path: {} not exists".format(match_rule))
            return

    session = cs.CreateSession()
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
            os.makedirs(os.path.dirname(CKPT_OUTPUT_PATH), mode=0o755)
        mox.file.copy_parallel(CKPT_OUTPUT_PATH, args.train_url)
        if args.freeze_pb:
            freeze_ckpt_to_pb(args)
    elif args.mode == 'evaluate':
        trainer.evaluate()
    elif args.mode == 'train_and_evaluate':
        trainer.train_and_evaluate()
        # 训练完成后把生成的模型拷贝到指导输出目录
        if not os.path.exists(os.path.dirname(CKPT_OUTPUT_PATH)):
            os.makedirs(os.path.dirname(CKPT_OUTPUT_PATH), mode=0o755)
        mox.file.copy_parallel(CKPT_OUTPUT_PATH, args.train_url)
    else:
        raise ValueError("Invalid mode.")


if __name__ == '__main__':
    main()

