#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Copyright 2021 Huawei Technologies Co., Ltd
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

'\nCreated on Feb 26 2017\nAuthor: Weiping Song\n'
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import model
import evaluation

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        # modify for npu op overflow start
        if command_line.over_dump is True:
            custom_op.parameter_map["enable_dump_debug"].b = True
            custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
            custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(command_line.over_dump_path)
        else:
            pass
        # modify for npu op overflow end
    return session_config
PATH_TO_TRAIN = './data/rsc15_train_full.txt'
PATH_TO_TEST = './data/rsc15_test.txt'

class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 4096
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = (- 1)


def broadcast_global_variables(root_rank, index):
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    return tf.group(op_list)


def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--npu_nums', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--path_to_train', default='./data/', type=str)
    # modify for npu op dump start
    parser.add_argument('--path_to_test', default='rsc15_test.txt', type=str)
    parser.add_argument('--train_dataset_file', default='/rsc15_train_full.txt', type=str)
    parser.add_argument('--over_dump', type=bool, default=False,
                        help='Whether to enable op overflow dump, default is False')
    parser.add_argument('--over_dump_path', type=str, default='/home/output/overflow_dump',
                        help='Directory name to save the overflow dump files')
    # modify for npu op dump end
    return parser.parse_args()
if (__name__ == '__main__'):
    command_line = parseArgs()
    data = pd.read_csv(command_line.path_to_train+command_line.train_dataset_file, sep='\t', dtype={'ItemId': np.int64})
    # modify for npu op dump start
    # valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(command_line.path_to_test+'/rsc15_test.txt', sep='\t', dtype={'ItemId': np.int64})
    # modify for npu op dump start
    args = Args()
    args.n_items = len(data['ItemId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.npu_nums = command_line.npu_nums
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = (1.0 if (args.is_training == 0) else command_line.dropout)
    # modify for npu op dump start
    args.over_dump = command_line.over_dump
    args.over_dump_path = command_line.over_dump_path
    # modify for npu op dump end
    print(args.dropout_p_hidden)
    if (not os.path.exists(args.checkpoint_dir)):
        os.mkdir(args.checkpoint_dir)
    if True:
        gpu_config = tf.ConfigProto()
        gpu_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    gpu_config.gpu_options.allow_growth = True
    if args.npu_nums == 8:
        bocast_op = broadcast_global_variables(0, 1)
    with tf.Session(config=npu_session_config_init(session_config=gpu_config)) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.npu_nums == 8:
            sess.run(bocast_op)
        if args.is_training:
            gru.fit(data)
        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))