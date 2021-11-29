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

'\nCreated on Feb 27 2017\nAuthor: Weiping Song\n'
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import sys
import numpy as np
import argparse
import tensorflow as tf
from model import GRU4Rec
from utils import load_test

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config
unfold_max = 20
cut_off = 20
(test_x, test_y, n_items) = load_test(unfold_max)

class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 10
    batch_size = 50
    keep_prob = 1
    learning_rate = 0.002
    decay = 0.98
    decay_steps = (1000.0 * 5)
    sigma = 0.0005
    init_as_normal = False
    grad_cap = 0
    test_model = 9
    checkpoint_dir = 'save/{}'.format('lstm')
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = (- 1)

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dr', default=0.98, type=float)
    parser.add_argument('--ds', default=400, type=int)
    parser.add_argument('--keep', default='1.0', type=float)
    command_line = parser.parse_args()
    args.layers = command_line.layer
    args.batch_size = command_line.batch
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.rnn_size = command_line.size
    args.keep_prob = command_line.keep
    args.decay = command_line.dr
    args.decay_steps = command_line.ds
    args.checkpoint_dir += ('_p' + str(command_line.keep))
    args.checkpoint_dir += ('_rnn' + str(command_line.size))
    args.checkpoint_dir += ('_batch' + str(command_line.batch))
    args.checkpoint_dir += ('_lr' + str(command_line.lr))
    args.checkpoint_dir += ('_dr' + str(command_line.dr))
    args.checkpoint_dir += ('_ds' + str(command_line.ds))
    args.checkpoint_dir += ('_unfold' + str(unfold_max))
    return args

def evaluate(args):
    '\n    Returns\n    --------\n    out : tuple\n        (Recall@N, MRR@N)\n    '
    args.n_items = n_items
    evaluation_point_count = 0
    (mrr_l, recall_l, ndcg20_l, ndcg_l) = (0.0, 0.0, 0.0, 0.0)
    np.random.seed(42)
    if True:
        gpu_config = tf.ConfigProto()
        gpu_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    gpu_config.gpu_options.allow_growth = True
    model = GRU4Rec(args)
    with tf.Session(config=npu_session_config_init(session_config=gpu_config)) as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if (ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model from {} successfully!'.format(args.checkpoint_dir))
        else:
            print('Restore model from {} failed!'.format(args.checkpoint_dir))
            return
        batch_idx = 0
        while (batch_idx < len(test_x)):
            batch_x = test_x[batch_idx:(batch_idx + args.batch_size)]
            batch_y = test_y[batch_idx:(batch_idx + args.batch_size)]
            feed_dict = {model.X: batch_x, model.Y: batch_y}
            (hit, ndcg, n_target) = sess.run([model.hit_at_k, model.ndcg_at_k, model.num_target], feed_dict=feed_dict)
            recall_l += hit
            ndcg_l += ndcg
            evaluation_point_count += n_target
            batch_idx += args.batch_size
    return ((recall_l / evaluation_point_count), (ndcg_l / evaluation_point_count))
if (__name__ == '__main__'):
    args = parseArgs()
    res = evaluate(args)
    print('lr: {}\tbatch_size: {}\tdecay_steps:{}\tdecay_rate:{}\tkeep_prob:{}\tdim: {}\tlayer: {}'.format(args.learning_rate, args.batch_size, args.decay_steps, args.decay, args.keep_prob, args.rnn_size, args.layers))
    print('Recall@20: {}\tNDCG: {}'.format(res[0], res[1]))
    sys.stdout.flush()
