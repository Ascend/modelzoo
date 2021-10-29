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
import glob
import os
import sys

import moxing as mox
import tensorflow as tf

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/..")
sys.path.append(base_path + "/../models")
sys.path.append(base_path + "/../../")
sys.path.append(base_path + "/../../src")

from utils import create_session as cs
from utils import logger as lg
from data_loader.resnet50 import data_loader as dl
from models.resnet50 import res50_model as ml
from model_transform import replace_ckpt_with_placeholder
from model_transform import frozen_graph
from optimizers import optimizer as op
from losses import res50_loss as ls
from trainers import gpu_base_trainer as tr
from hyper_param import hyper_param as hp
from layers import layers as ly


CKPT_OUTPUT_PATH = "/cache/ckpt_first"
PB_OUTPUT_PATH = "/cache/model"


def set_env():
    os.environ['DEVICE_INDEX'] = os.getenv('RANK_ID')


def args_parser():
    parser = argparse.ArgumentParser(
        description="train resnet50",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    # 数据集目录
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    # 使用的参数配置文件
    parser.add_argument('--config_file', type=str, default='res50_256bs_1p',
                        help='the config file')

    # 抽取出来的超参设置
    parser.add_argument('--max_train_steps', type=int, default=1000,
                        help='max_train_steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='num_classes')
    # 指定要训练多少个epoch，如果该值非None，则参数max_train_steps无效
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='num_epochs')
    parser.add_argument('--learning_rate_maximum', type=float, default=0.1,
                        help='learning_rate_maximum')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='whether to do model evaluation')
    parser.add_argument('--over_dump', type=bool, default=False,
                        help='whether to do model evaluation')
    parser_args, _ = parser.parse_known_args()
    return parser_args


def set_config(args):
    configs = 'configs'
    cfg = getattr(__import__(configs, fromlist=[args.config_file]),
                  args.config_file)
    config = cfg.res50_config()
    # 默认参数设置
    config['iterations_per_loop'] = 10
    config['debug'] = False
    # ModelArts数据集存储路径
    config['data_url'] = "/cache"
    config['log_dir'] = '/cache/ckpt_first'
    config['model_dir'] = "/cache/ckpt_first"

    # 设置传入的超参数
    config['max_train_steps'] = args.max_train_steps
    config['batch_size'] = args.batch_size
    config['global_batch_size'] = config['batch_size'] * config['rank_size']
    config['num_classes'] = args.num_classes
    config['num_epochs'] = args.num_epochs
    config['learning_rate_maximum'] = args.learning_rate_maximum
    config['eval'] = args.do_eval
    config["over_dump"] = args.over_dump

    print("max_train_steps        :%d" % (config['max_train_steps']))
    print("batch_size             :%d" % (config['batch_size']))
    print("num_classes            :%d" % (config['num_classes']))
    if config['num_epochs']:
        print("num_epochs         :%d" % (config['num_epochs']))
    print("learning_rate_maximum  :%d" % (config['learning_rate_maximum']))

    return config


def train(args):
    # 设置config配置文件相关参数
    config = set_config(args)

    # 设置模型参数
    session = cs.CreateSession(config)
    data = dl.DataLoader(config)
    hyper_param = hp.HyperParams(config)
    layers = ly.Layers()
    optimizer = op.Optimizer(config)
    loss = ls.Loss(config)
    logger = lg.LogSessionRunHook(config)

    # 使用Estimator来构建训练流程
    model = ml.Model(config, data, hyper_param, layers, optimizer, loss, logger)
    trainer = tr.GPUBaseTrain(session, config, data, model, logger)

    if config['mode'] == 'train':
        trainer.train()
        if config['eval']:
            trainer.evaluate()
    elif config['mode'] == 'evaluate':
        trainer.evaluate()
    elif config['mode'] == 'train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError('Invalid type of mode')


def model_trans():
    # 占位转换
    origin_ckpt_path = "/cache/ckpt_first"
    placed_ckpt_path = "/cache/ckpt_first/placeholder"
    pb_path = "/cache/model/resnet_placeholder.pb"
    match_rule = "model.ckpt-[!0]*.meta*"
    ckpt_match_path = os.path.join(origin_ckpt_path, match_rule)
    ckpt_list = glob.glob(ckpt_match_path)
    if not ckpt_list:
        print("ckpt file not generated.")
        return
    ckpt_list.sort(key=lambda fn: os.path.getmtime(fn))
    ckpt_model = ckpt_list[-1].rsplit(".", 1)[0]
    placeholder_path = os.path.join(placed_ckpt_path,
                                    ckpt_model.rsplit("/", 1)[1])
    replace_ckpt_with_placeholder.add_placeholder_on_ckpt(
        ckpt_model, placeholder_path)

    if not os.path.exists(PB_OUTPUT_PATH):
        os.makedirs(PB_OUTPUT_PATH, exist_ok=True)

    # 设置模型参数
    placed_match_path = os.path.join(placed_ckpt_path, match_rule)
    placed_ckpt_list = glob.glob(placed_match_path)
    if not placed_ckpt_list:
        print("after placeholder, ckpt file not generated.")
        return
    placed_ckpt_list.sort(key=lambda fn: os.path.getmtime(fn))
    ckpt = placed_ckpt_list[-1].rsplit(".", 1)[0]
    output_nodes = 'fp32_vars/final_dense'
    model_input = {'meta_file': ckpt,
                   'output_file': pb_path,
                   'output_nodes': output_nodes}
    # 转换ckpt模型为pb格式
    frozen_graph.main(model_input)
    # 拷贝pb模型到模型输出目录
    mox.file.copy_parallel(PB_OUTPUT_PATH, input_args.train_url)


if __name__ == '__main__':
    # 设置环境变量
    set_env()

    # 解析传入参数
    input_args = args_parser()

    # 现将数据集拷贝到ModelArts指定读取的cache目录
    mox.file.copy_parallel(input_args.data_url, '/cache')

    # 设置日志打印级别
    tf.logging.set_verbosity(tf.logging.INFO)

    # 开始训练
    train(input_args)

    # 训练完成后把生成的模型拷贝到指导输出目录
    if not os.path.exists(os.path.dirname(CKPT_OUTPUT_PATH)):
        os.makedirs(os.path.dirname(CKPT_OUTPUT_PATH), exist_ok=True)
    mox.file.copy_parallel(CKPT_OUTPUT_PATH, input_args.train_url)

    # 转换ckpt模型为pb格式，可选
    model_trans()

