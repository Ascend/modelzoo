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
"""DeepFM related model"""
from __future__ import print_function

import os
import sys
import pickle
import tensorflow as tf
from deepfm.tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
rank_size = int(os.getenv('RANK_SIZE'))

class FMNN_v2:
    """support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """
    def __init__(self,  dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, _input_data, loss_mode='full', merge_multi_hot=False,
                 cross_layer=False, batch_norm=False):
        # self.graph = tf.Graph()
        #with tf.variable_scope('FMNNMG') as scope:
        (input_dim, input_dim4lookup,
         self.multi_hot_flags, self.multi_hot_len) = dataset_argv
        self.one_hot_flags = [not flag for flag in self.multi_hot_flags]
        embed_size, layer_dims, act_func = architect_argv
        keep_prob, _lambda = reg_argv
        self.num_onehot = sum(self.one_hot_flags)
        self.num_multihot = sum(self.multi_hot_flags) / self.multi_hot_len
        if merge_multi_hot:
            self.embed_dim = (self.num_multihot +
                              self.num_onehot) * embed_size
        else:
            self.embed_dim = input_dim4lookup * embed_size
        self.all_layer_dims = [self.embed_dim + 1] + layer_dims + [1]
        self.log = ('input dim: %d\nnum inputs: %d\nembed size(each): %d\n'
                    'embedding layer: %d\nlayers: %s\nactivate: %s\n'
                    'keep_prob: %g\nl2(lambda): %g\nmerge_multi_hot: %s\n' %
                    (input_dim, input_dim4lookup, embed_size,
                     self.embed_dim, self.all_layer_dims, act_func,
                     keep_prob, _lambda, merge_multi_hot))

        self.fm_v = tf.get_variable('V', shape=[input_dim, embed_size],
                                    initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    dtype=tf.float32)
        self.fm_w = tf.get_variable('W', shape=[input_dim, 1],
                                    initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    dtype=tf.float32)
        self.fm_b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer, dtype=tf.float32)

        self.h_w, self.h_b = [], []
        for i in range(len(self.all_layer_dims) - 1):
            self.h_w.append(tf.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                            initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                            dtype=tf.float32))
            self.h_b.append(
                tf.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]], initializer=tf.zeros_initializer,
                                dtype=tf.float32))

        # self.wt_hldr = tf.placeholder(tf.float32,
        #                               shape=[None, input_dim4lookup])
        # self.id_hldr = tf.placeholder(tf.int64,
        #                               shape=[None, input_dim4lookup])
        # self.lbl_hldr = tf.placeholder(tf.float32)

        self.wt_hldr = _input_data[2]
        self.id_hldr = _input_data[1]
        self.lbl_hldr = _input_data[0]

        logits, wx, vx_embed = self.forward(
            self.wt_hldr, self.id_hldr, act_func, keep_prob, training=True,
            merge_multi_hot=merge_multi_hot, cross_layer=cross_layer,
            batch_norm=batch_norm)
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self.lbl_hldr)
        self.log_loss = tf.reduce_mean(log_loss)
        self.train_preds = tf.sigmoid(logits, name='predictions')
        if loss_mode == 'full':
            self.l2_loss = _lambda * (tf.nn.l2_loss(self.fm_w) +
                                      tf.nn.l2_loss(self.fm_v))
        else:  # 'batch'
            self.l2_loss = _lambda * (tf.nn.l2_loss(wx) +
                                      tf.nn.l2_loss(vx_embed))
        self.loss = self.log_loss + self.l2_loss
        self.eval_wt_hldr = tf.placeholder(
            tf.float32, shape=[None, input_dim4lookup], name='wt')
        self.eval_id_hldr = tf.placeholder(
            tf.int64, shape=[None, input_dim4lookup], name='id')
        eval_logits, _, _ = self.forward(
            self.eval_wt_hldr, self.eval_id_hldr, act_func, keep_prob,
            training=False, merge_multi_hot=merge_multi_hot,
            cross_layer=cross_layer, batch_norm=batch_norm)
        self.eval_preds = tf.sigmoid(eval_logits, name='predictionNode')

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.ptmzr, log = build_optimizer(ptmzr_argv, self.loss)
        # self.log += log




    def forward(self, wt_hldr, id_hldr, act_func, keep_prob,
                training, merge_multi_hot=False,
                cross_layer=False, batch_norm=False):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            one_hot_mask, multi_hot_mask = split_mask(
                mask, self.multi_hot_flags, self.num_multihot)
            one_hot_w, multi_hot_w = split_param(
                self.fm_w, id_hldr, self.multi_hot_flags)
            one_hot_v, multi_hot_v = split_param(
                self.fm_v, id_hldr, self.multi_hot_flags)

            # linear part
            multi_hot_wx = sum_multi_hot(
                multi_hot_w, multi_hot_mask, self.num_multihot)
            one_hot_wx = tf.multiply(one_hot_w, one_hot_mask)
            wx = tf.concat([one_hot_wx, multi_hot_wx], axis=1)

            # fm part (reduce multi-hot vector's length to k*1)
            multi_hot_vx = sum_multi_hot(
                multi_hot_v, multi_hot_mask, self.num_multihot)
            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            # [batch, input_dim4lookup, 1]
            wx = tf.multiply(tf.gather(self.fm_w, id_hldr), mask)
            # [batch, input_dim4lookup, embed_size]
            vx_embed = tf.multiply(tf.gather(self.fm_v, id_hldr), mask)

        linear_out = tf.reduce_sum(wx, 1)
        b = tf.reshape(tf.tile(tf.identity(self.fm_b),
                               tf.shape(wt_hldr)[0:1]),
                       [-1, 1])
        v2x2 = tf.square(vx_embed)
        vx2 = tf.square(tf.reduce_sum(vx_embed, 1))
        vxs = tf.reduce_sum(v2x2, 1)
        fm_out = 0.5 * tf.reduce_sum(vx2 - vxs, 1)

        hidden_output = tf.concat(
            [tf.reshape(vx_embed, [-1, self.embed_dim]), b], axis=1)
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(
                    npu_ops.dropout(
                        activate(act_func, hidden_output), keep_prob=keep_prob),
                    self.h_w[i]) + self.h_b[i]
            else:
                hidden_output = tf.matmul(
                    activate(act_func, hidden_output),
                    self.h_w[i]) + self.h_b[i]
            # if batch_norm:
            #     hidden_output = tf.layers.batch_normalization(
            #         hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output

        return tf.reshape(hidden_output, [-1, ]) + \
            tf.reshape(fm_out, [-1, ]) + \
            tf.reshape(linear_out, [-1, ]), wx, vx_embed

    def dump(self, model_path):
        var_map = {'W': self.fm_w.eval(),
                   'V': self.fm_v.eval(),
                   'b': self.fm_b.eval()}

        for i, (h_w_i, h_b_i) in enumerate(zip(self.h_w, self.h_b)):
            var_map['h%d_w' % (i+1)] = h_w_i.eval()
            var_map['h%d_b' % (i+1)] = h_b_i.eval()

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)


class FMNN:
    def __init__(self,  _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, mode, src, loss_mode='full'):
        self.src = src
        self.graph = tf.Graph()
        with self.graph.as_default():
            X_dim, M, rank, self.l_dims, act_func = _rch_argv
            K = rank
            mbd_dim = M * K + 1
            self.log = 'input dim: %d, features: %d, rank: %d, embedding: %d, self.l_dims: %s, activate: %s, ' % \
                       (X_dim, M, rank, mbd_dim, str(self.l_dims), act_func)

            init_acts = [('W', [X_dim, 1], 'random'),
                         ('V', [X_dim, rank], 'random'),
                         ('b', [1], 'zero')]
            if len(self.l_dims) == 1:
                h1_dim = self.l_dims[0]
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, 1], 'random'),
                                  ('h2_b', [1], 'zero'), ])
            elif len(self.l_dims) == 3:
                h1_dim, h2_dim, h3_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, 1], 'random'),
                                  ('h4_b', [1], 'zero')])
            elif len(self.l_dims) == 5:
                h1_dim, h2_dim, h3_dim, h4_dim, h5_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, h4_dim], 'random'),
                                  ('h4_b', [h4_dim], 'zero'),
                                  ('h5_w', [h4_dim, h5_dim], 'random'),
                                  ('h5_b', [h5_dim], 'zero'),
                                  ('h6_w', [h5_dim, 1], 'random'),
                                  ('h6_b', [1], 'zero')])
            elif len(self.l_dims) == 7:
                h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, h6_dim, h7_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, h4_dim], 'random'),
                                  ('h4_b', [h4_dim], 'zero'),
                                  ('h5_w', [h4_dim, h5_dim], 'random'),
                                  ('h5_b', [h5_dim], 'zero'),
                                  ('h6_w', [h5_dim, h6_dim], 'random'),
                                  ('h6_b', [h6_dim], 'zero'),
                                  ('h7_w', [h6_dim, h7_dim], 'random'),
                                  ('h7_b', [h7_dim], 'zero'),
                                  ('h8_w', [h7_dim, 1], 'random'),
                                  ('h8_b', [1], 'zero')])

            var_map, log = init_var_map(_init_argv, init_acts)

            self.log += log
            self.fm_w = tf.get_variable(var_map['W'])
            self.fm_v = tf.get_variable(var_map['V'])
            self.fm_b = tf.get_variable(var_map['b'])
            if len(self.l_dims) == 1:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
            elif len(self.l_dims) == 3:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
            elif len(self.l_dims) == 5:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
                self.h5_w = tf.get_variable(var_map['h5_w'])
                self.h5_b = tf.get_variable(var_map['h5_b'])
                self.h6_w = tf.get_variable(var_map['h6_w'])
                self.h6_b = tf.get_variable(var_map['h6_b'])
            elif len(self.l_dims) == 7:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
                self.h5_w = tf.get_variable(var_map['h5_w'])
                self.h5_b = tf.get_variable(var_map['h5_b'])
                self.h6_w = tf.get_variable(var_map['h6_w'])
                self.h6_b = tf.get_variable(var_map['h6_b'])
                self.h7_w = tf.get_variable(var_map['h7_w'])
                self.h7_b = tf.get_variable(var_map['h7_b'])
                self.h8_w = tf.get_variable(var_map['h8_w'])
                self.h8_b = tf.get_variable(var_map['h8_b'])

            self.wt_hldr = tf.placeholder(tf.float32, shape=[None, M])
            self.id_hldr = tf.placeholder(tf.int64, shape=[None, M])
            self.lbl_hldr = tf.placeholder(tf.float32)

            self.fm_wv = tf.concat([self.fm_w, self.fm_v], 1)

            keep_prob = _reg_argv[0]
            _lambda = _reg_argv[1]
            if mode == 'train':
                logits = self.forward(K, M, self.wt_hldr, self.id_hldr, act_func, keep_prob, True)
                log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.lbl_hldr)
                if _ptmzr_argv[-1] == 'sum':
                    self.loss = tf.reduce_sum(log_loss)
                else:
                    self.loss = tf.reduce_mean(log_loss)
                self.train_preds = tf.sigmoid(logits, name='predictions')
                if loss_mode == 'full':
                    self.loss += _lambda * (tf.nn.l2_loss(self.fm_w) + tf.nn.l2_loss(self.fm_v))
                else: # 'batch'
                    self.loss += _lambda * (tf.nn.l2_loss(self.wx) + tf.nn.l2_loss(self.ori_z))
                self.ptmzr, log = build_optimizer(_ptmzr_argv, self.loss)
                self.log += '%s, reduce by: %s\tkeep_prob: %g' % (log, _ptmzr_argv[-1], keep_prob)

                self.eval_wt_hldr = tf.placeholder(tf.float32, shape=[None, M], name='wt')
                self.eval_id_hldr = tf.placeholder(tf.int64, shape=[None, M], name='id')

                eval_logits = self.forward(K, M, self.eval_wt_hldr, self.eval_id_hldr, act_func, keep_prob,
                                           False)
                self.eval_preds = tf.sigmoid(eval_logits, name='predictionNode')
            else:
                logits = self.forward(K, M, self.wt_hldr, self.id_hldr, act_func, keep_prob, False)
                self.test_preds = tf.sigmoid(logits)

    def forward(self, K, M, v_wts, c_ids, act_func, keep_prob, drop_out=False):
        mask = tf.reshape(v_wts, [-1, M, 1])
        self.wx = tf.multiply(tf.gather(self.fm_w, c_ids), mask)
        print("fm_w {}\nc_ids {}\nafter gather {}".format(self.fm_w.get_shape(), c_ids.get_shape(), tf.gather(self.fm_w, c_ids).get_shape()))

        if self.src == 'criteo':
            v_mbd = tf.reshape(v_wts[:, :13], [-1, 13, 1]) * tf.reshape(self.fm_v[:13, :], [1, 13, K])
            c_mbd = tf.gather(self.fm_v, c_ids[:, 13:])
            #b = tf.reshape(tf.concat([tf.identity(self.fm_b) for _i in range(N)], 0), [-1, 1])
            b = tf.reshape(tf.tile(tf.identity(self.fm_b), tf.shape(v_wts)[0:1]), [-1, 1])
            self.ori_z = tf.concat([v_mbd, c_mbd], 1)
        elif self.src == 'ipinyou':
            mask = tf.concat([tf.reshape(v_wts, [-1, M, 1]) for _i in range(K)], 2)
            c_mbd = tf.multiply(tf.gather(self.fm_v, c_ids), mask)
            #b = tf.reshape(tf.concat([tf.identity(self.fm_b) for _i in range(N)], 0), [N, 1])
            b = tf.reshape(tf.tile(tf.identity(self.fm_b), tf.shape(v_wts)[0:1]), [-1, 1])
            self.ori_z = c_mbd
        z = tf.transpose(self.ori_z, [0, 2, 1])

        v2x2 = tf.square(self.ori_z)
        vx2 = tf.square(tf.reduce_sum(self.ori_z,1))
        vxs = tf.reduce_sum(v2x2, 1)
        fmloss = 0.5 * tf.reduce_sum(vx2 - vxs, 1)
        if drop_out:
            l2 = tf.matmul(
                tf.nn.dropout(activate(act_func, tf.concat([tf.reshape(z, [-1, K*M]), b], 1)), keep_prob=keep_prob),
                self.h1_w) + self.h1_b
                #tf.nn.dropout(activate(act_func, tf.reshape(z, [-1, K*M])), keep_prob=keep_prob),
                #self.h1_w) + self.h1_b
            if len(self.l_dims) == 1:
                yhat = tf.matmul(tf.nn.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
            elif len(self.l_dims) == 3:
                l3 = tf.matmul(tf.nn.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(tf.nn.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                yhat = tf.matmul(tf.nn.dropout(activate(act_func, l4), keep_prob=keep_prob),
                                 self.h4_w) + self.h4_b
            elif len(self.l_dims) == 5:
                l3 = tf.matmul(tf.nn.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(tf.nn.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                l5 = tf.matmul(tf.nn.dropout(activate(act_func, l4), keep_prob=keep_prob), self.h4_w) + self.h4_b
                l6 = tf.matmul(tf.nn.dropout(activate(act_func, l5), keep_prob=keep_prob), self.h5_w) + self.h5_b
                yhat = tf.matmul(tf.nn.dropout(activate(act_func, l6), keep_prob=keep_prob),
                                 self.h6_w) + self.h6_b
            elif len(self.l_dims) == 7:
                l3 = tf.matmul(tf.nn.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(tf.nn.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                l5 = tf.matmul(tf.nn.dropout(activate(act_func, l4), keep_prob=keep_prob), self.h4_w) + self.h4_b
                l6 = tf.matmul(tf.nn.dropout(activate(act_func, l5), keep_prob=keep_prob), self.h5_w) + self.h5_b
                l7 = tf.matmul(tf.nn.dropout(activate(act_func, l6), keep_prob=keep_prob), self.h6_w) + self.h6_b
                l8 = tf.matmul(tf.nn.dropout(activate(act_func, l7), keep_prob=keep_prob), self.h7_w) + self.h7_b
                yhat = tf.matmul(tf.nn.dropout(activate(act_func, l8), keep_prob=keep_prob),
                                 self.h8_w) + self.h8_b
        else:
            #l2 = tf.matmul(tf.nn.dropout(activate(act_func, tf.reshape(z, [-1, M*K])), 1), self.h1_w) + self.h1_b
            l2 = tf.matmul(activate(act_func, tf.concat([tf.reshape(z, [-1, K*M]), b], 1)), self.h1_w) + self.h1_b

            if len(self.l_dims) == 1:
                yhat = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
            elif len(self.l_dims) == 3:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                yhat = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
            elif len(self.l_dims) == 5:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                l5 = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
                l6 = tf.matmul(activate(act_func, l5), self.h5_w) + self.h5_b
                yhat = tf.matmul(activate(act_func, l6), self.h6_w) + self.h6_b
            elif len(self.l_dims) == 7:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                l5 = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
                l6 = tf.matmul(activate(act_func, l5), self.h5_w) + self.h5_b
                l7 = tf.matmul(activate(act_func, l6), self.h6_w) + self.h6_b
                l8 = tf.matmul(activate(act_func, l7), self.h7_w) + self.h7_b
                yhat = tf.matmul(activate(act_func, l8), self.h8_w) + self.h8_b
        #return tf.reshape(tf.add(yhat, fmloss), [-1, ])
        return tf.reshape(yhat,[-1, ]) + tf.reshape(fmloss, [-1, ]) + tf.reshape(tf.reduce_sum(self.wx, 1), [-1, ])

    def dump(self, model_path):
        if len(self.l_dims) == 1:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval()}
        elif len(self.l_dims) == 3:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval()}
        elif len(self.l_dims) == 5:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval(), 'h5_w': self.h5_w.eval(), 'h5_b': self.h5_b.eval(),
                       'h6_w': self.h6_w.eval(), 'h6_b': self.h6_b.eval()}
        elif len(self.l_dims) == 7:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval(), 'h5_w': self.h5_w.eval(), 'h5_b': self.h5_b.eval(),
                       'h6_w': self.h6_w.eval(), 'h6_b': self.h6_b.eval(), 'h7_w': self.h7_w.eval(),
                       'h7_b': self.h7_b.eval(), 'h8_w': self.h8_w.eval(), 'h8_b': self.h8_b.eval()}

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)
