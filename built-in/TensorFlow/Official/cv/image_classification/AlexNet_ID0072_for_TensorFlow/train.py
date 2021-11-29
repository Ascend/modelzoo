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
import tensorflow as tf
import numpy as np

import sys
import ast

import alexnet.data_loader as dl
import alexnet.model as ml
import alexnet.hyper_param as hp
import alexnet.layers as ly
import alexnet.logger as lg
import alexnet.trainer as tr
import alexnet.create_session as cs

import argparse


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--iterations_per_loop', default=1251, type=int,
                        help="""the number of steps in devices for each iteration""")
    parser.add_argument('--rank_size', default=1,type=int,
                        help="""number of NPUs  to use.""")
    parser.add_argument('--shard',  default=False, type=ast.literal_eval,
                        help="""whether to use shard or not""")
    parser.add_argument('--mode', default='train_and_evaluate',
                        help="""mode to run the program  e.g. train, evaluate, and 
                        train_and_evaluate """)
    parser.add_argument('--epochs_between_evals', default=5,type=int,
                        help="""the interval between train and evaluation , only meaningful 
                        when the mode is train_and_evaluate """)
    parser.add_argument('--data_dir', default='path/to/data',
                        help="""directory to data.""")
    parser.add_argument('--dtype', default=tf.float32,
                        help="""data type of inputs.""")
    parser.add_argument('--use_nesterov', default=True, type=ast.literal_eval,
                        help=""" used in optimizer""")
    parser.add_argument('--label_smoothing', default=0.1,type=float,
                        help="""label smoothing factor""")
    parser.add_argument('--weight_decay', default=0.0001,
                        help="""weight decay""")
    parser.add_argument('--batch_size', default=128,type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--max_train_steps', default=None,type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--warmup_epochs', default=5,type=int,
                        help="""number of warmup epochs """)
    parser.add_argument('--lr', default=0.06,type=float,
                        help="""learning rate""")
    parser.add_argument('--max_epochs', default=150,type=int,
                        help="""total epochs for training""")
    parser.add_argument('--momentum', default=0.9,type=float,
                        help="""momentum used in optimizer.""")

    parser.add_argument('--synthetic', default=False,type=ast.literal_eval,
                        help="""parser file used.""")

    parser.add_argument('--debug', default=True, type=ast.literal_eval,
                         help="""parser file used.""")
    parser.add_argument('--eval', default=False, type=ast.literal_eval,
                         help="""parser file used.""")

    parser.add_argument('--display_every', default=1,type=int,
                        help="""the frequency to display info""")
    parser.add_argument('--log_name', default='alexnet_training.log',
                        help="""name of log file""")
    parser.add_argument('--alexnet_version', default='he_uniform',
                        help="""version of weight initialization""")

    parser.add_argument('--log_dir', default='./model_1p',
                        help="""log directory""")
    parser.add_argument('--save_summary_steps', default=100,type=int,
                        help="""frequency to save summary""")
    parser.add_argument('--max_checkpoint_to_save', default=5,type=int,
                        help="""frequency to save checkpoints""")
    parser.add_argument('--checkpoint_dir', default='./model_8p',
                         help="""directory to checkpoints""")

    parser.add_argument('--num_classes', default=1000,type=int,
                        help="""number of classes for datasets """)
    parser.add_argument('--save_checkpoints_steps', default=1000,type=int,
                        help="""frequency to save checkpoints""")
    parser.add_argument('--display', default=1,type=int,
                        help="""the frequency to display info""")
    parser.add_argument('--do_checkpoint', default=True, type=ast.literal_eval,
                        help="""whether to save checkpoints or not.""")
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

def main():

    args = parse_args()
    args.global_batch_size = args.batch_size *args.rank_size

    Session = cs.CreateSession(args)
    data = dl.DataLoader(args)
    hyper_param = hp.HyperParams(args)
    layers = ly.Layers()
    logger = lg.LogSessionRunHook(args)
    model = ml.Model(args, data, hyper_param, layers, logger)
    trainer = tr.Trainer(Session, args, data, model, logger)

    if args.mode =='train':  
        trainer.train()
    elif args.mode =='evaluate':
        trainer.evaluate()
    else:
        raise ValueError('Invalid type of mode')


if __name__ == '__main__':
    main()

