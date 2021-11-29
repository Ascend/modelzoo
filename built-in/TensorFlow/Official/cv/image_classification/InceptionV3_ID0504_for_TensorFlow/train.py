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

import inception.data_loader as dl
import inception.model as ml
import inception.hyper_param as hp
import inception.layers as ly
import inception.logger as lg
import inception.trainer as tr
import inception.create_session as cs

import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

    parser.add_argument('--rank_size', default=1,type=int,
                        help="""number of NPUs  to use.""")
    parser.add_argument('--network', default = 'inception_v3', type=str,
                        help = """the network name""")

    # mode and parameters related 
    parser.add_argument('--mode', default='train',
                        help="""mode to run the program  e.g. train, evaluate, and
                        train_and_evaluate""")
    parser.add_argument('--max_train_steps', default=100,type=int,
                        help="""max steps to train""")
    parser.add_argument('--iterations_per_loop', default=10, type=int,
                        help="""the number of steps in devices for each iteration""")
    parser.add_argument('--max_epochs', default=100, type=int,
                        help="""total epochs for training""")
    parser.add_argument('--epochs_between_evals', default=5, type=int,
                        help="""the interval between train and evaluation , only meaningful
                        when the mode is train_and_evaluate""")

    # dataset
    parser.add_argument('--data_dir', default='path/data',
                        help="""directory of dataset.""")

    # path for evaluation
    parser.add_argument('--eval_dir', default='./model',
                        help="""directory to evaluate.""")

    parser.add_argument('--dtype', default=tf.float32,
                        help="""data type of inputs.""")
    parser.add_argument('--use_nesterov', default=True, type=ast.literal_eval,
                        help="""whether to use Nesterov in optimizer""")
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help="""label smoothing factor""")
    parser.add_argument('--weight_decay', default=0.00001,
                        help="""weight decay for regularization""")
    parser.add_argument('--batch_size', default=256, type=int,
                        help="""batch size for one NPU""")

    # learning rate and momentum
    parser.add_argument('--lr', default=0.045, type=float,
                        help="""initial learning rate""")
    parser.add_argument('--lr_decay', default = 0.8, type = float,
                        help = """learning rate decay""")
    parser.add_argument("--lr_decay_steps", default = 10000, type = int,
                        help = """learning rate decay steps""")
    parser.add_argument('--T_max', default=100, type=int,
                        help="""T_max for cosing_annealing learning rate""")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="""momentum used in optimizer.""")

    # display frequency
    parser.add_argument('--display_every', default=100, type=int,
                        help="""the frequency to display info""")

    # log file
    parser.add_argument('--log_name', default='inception_v3.log',
                        help="""name of log file""")
    parser.add_argument('--log_dir', default='./model',
                        help="""log directory""")

    parser.add_argument('--num_classes', default=1000, type=int,
                        help="the number class of dataset")

    # pre_train model path
    parser.add_argument("--restore_path", default=None,
                        help="restore path")
    # modify for npu overflow start
    # enable overflow
    parser.add_argument("--over_dump", default=None,
                        help="whether to enable overflow")
    parser.add_argument("--over_dump_path", default=None,
                        help="path to save overflow dump files")
    # modify for npu overflow end
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def main():

    args = parse_args()
    args.global_batch_size = args.batch_size * args.rank_size

    session = cs.CreateSession()
    data = dl.DataLoader(args)    
    hyper_param = hp.HyperParams(args)
    layers = ly.Layers() 
    logger = lg.LogSessionRunHook(args)
    model = ml.Model(args, data, hyper_param, layers, logger)
   
    trainer = tr.Trainer(session, args, data, model, logger)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        trainer.evaluate()
    elif args.mode == 'train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError("Invalid mode.")


if __name__ == '__main__':
    main()

