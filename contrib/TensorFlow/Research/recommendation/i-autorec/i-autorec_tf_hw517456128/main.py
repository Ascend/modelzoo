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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import time
import argparse
import npu_bridge
import tensorflow as tf

from data_preprocessor import *
from AutoRec import AutoRec
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

parser = argparse.ArgumentParser(description='I-AutoRec')
parser.add_argument('--hidden_neuron', type=int, default=1024)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--optimizer_method', choices=['Adam','RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--data_url', type=str, default='./data')
parser.add_argument('--train_url', type=str, default='./output')

parser.add_argument('--data_name', type=str, default='ml-1m')
parser.add_argument('--num_users', type=int, default=6040)
parser.add_argument('--num_items', type=int, default=3952)
parser.add_argument('--num_total_ratings', type=int, default=1000209)
parser.add_argument('--train_ratio', type=float, default=0.9)


args = parser.parse_args()
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)


result_path = args.train_url + args.data_name + '/' + str(args.random_seed) + '_' + str(args.optimizer_method) + '_' + str(args.base_lr) + "_" + str(time.time())+"/"
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(args.data_url, args.num_users, args.num_items, args.num_total_ratings, 1, 0, args.train_ratio)


config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF


with tf.Session(config=config) as sess:
    AutoRec = AutoRec(sess,args,
                      args.num_users, args.num_items,
                      R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                      user_train_set, item_train_set, user_test_set, item_test_set,
                      result_path)
    AutoRec.run()
