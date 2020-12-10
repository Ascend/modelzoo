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
from widedeep.tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.hccl import hccl_ops
#from mindspore.ops import operations as P

rank_size = int(os.getenv('RANK_SIZE'))

class WideDeep:
    """support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """
    def __init__(self, graph, dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, _input_data, loss_mode='full', merge_multi_hot=False,
                 cross_layer=False, batch_norm=False):
        self.graph = graph
        #self.MEloss = P.SigmoidCrossEntropyWithLogits()
        with self.graph.as_default():
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
            self.all_layer_dims = [self.embed_dim] + layer_dims + [1]
            self.log = ('input dim: %d\nnum inputs: %d\nembed size(each): %d\n'
                        'embedding layer: %d\nlayers: %s\nactivate: %s\n'
                        'keep_prob: %g\nl2(lambda): %g\nmerge_multi_hot: %s\n' %
                        (input_dim, input_dim4lookup, embed_size,
                         self.embed_dim, self.all_layer_dims, act_func,
                         keep_prob, _lambda, merge_multi_hot))
            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                self.embed_v = tf.get_variable('V', shape=[input_dim, embed_size],
                                            initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                            dtype=tf.float32,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "embedding"])
            with tf.variable_scope("wide", reuse=tf.AUTO_REUSE):
                self.wide_w = tf.get_variable('wide_w', shape=[input_dim, 1],
                                            initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                            dtype=tf.float32,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, "wide", "wide_w"])
                self.wide_b = tf.get_variable('wide_b', [1],
                                            initializer = tf.random_uniform_initializer(-0.01, 0.01),
                                            dtype=tf.float32,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, "wide", "wide_b"])


            with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
                self.h_w, self.h_b = [], []
                for i in range(len(self.all_layer_dims) - 1):
                    self.h_w.append(tf.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                    dtype=tf.float32,
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"]))
                    self.h_b.append(
                        tf.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]], initializer=tf.zeros_initializer,
                                        dtype=tf.float32,
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"]))

            self.wt_hldr = _input_data[2]
            self.id_hldr = _input_data[1]
            self.lbl_hldr = _input_data[0]

            print("_input_data[0]== self.lbl_hldr", self.lbl_hldr)

            wideout = self.wide_forward(self.wt_hldr, self.id_hldr)
            y, vx_embeds = self.forward(
                self.wt_hldr, self.id_hldr, act_func, keep_prob, training=True,
                merge_multi_hot=merge_multi_hot, cross_layer=cross_layer,
                batch_norm=batch_norm)
            y = y + wideout
            self.train_preds = tf.sigmoid(y, name='predicitons')

            basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=self.lbl_hldr)

            self.wide_loss = tf.reduce_mean(basic_loss)
            self.deep_loss = tf.reduce_mean(basic_loss) + _lambda * tf.nn.l2_loss(self.embed_v)

            self.l2_loss = tf.constant([0]) #self.loss
            self.log_loss = basic_loss

            self.eval_wt_hldr = tf.placeholder(tf.float32, [None, input_dim4lookup], name='wt')
            self.eval_id_hldr = tf.placeholder(tf.int32, [None, input_dim4lookup], name='id')

            eval_wideout= self.wide_forward(self.eval_wt_hldr, self.eval_id_hldr)
            eval_y, eval_vx_embed = self.forward(
                self.eval_wt_hldr, self.eval_id_hldr, act_func, keep_prob, training=False,
                merge_multi_hot=merge_multi_hot, cross_layer=cross_layer,
                batch_norm=batch_norm)
            eval_y = eval_y + eval_wideout

            self.eval_preds = tf.sigmoid(eval_y, name='predictionNode')

            opt_deep, lr_deep, eps_deep, decay_rate_deep, decay_step_deep = ptmzr_argv[0]
            opt_wide, wide_lr, wide_dc, wide_l1, wide_l2 = ptmzr_argv[1]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                loss_scale_manager = FixedLossScaleManager(loss_scale=1000.0)
                loss_scale_manager2 = FixedLossScaleManager(loss_scale=1000.0)
                self.wide_ptmzr = tf.train.FtrlOptimizer(learning_rate=wide_lr, initial_accumulator_value=wide_dc,
                                                    l1_regularization_strength=wide_l1,
                                                    l2_regularization_strength=wide_l2)
                if rank_size > 1:
                    self.wide_ptmzr = NPULossScaleOptimizer(self.wide_ptmzr, loss_scale_manager, is_distributed=True)
                    grads_w = self.wide_ptmzr.compute_gradients(self.wide_loss, var_list=tf.get_collection('wide'))
                else:
                    self.wide_ptmzr = NPULossScaleOptimizer(self.wide_ptmzr, loss_scale_manager)
                    self.wide_ptmzr = self.wide_ptmzr.minimize(self.wide_loss,var_list=tf.get_collection("wide"))
                self.deep_optimzer = tf.train.AdamOptimizer(learning_rate=lr_deep, epsilon=eps_deep)
                if rank_size > 1:
                   self.deep_optimzer = NPULossScaleOptimizer(self.deep_optimzer, loss_scale_manager2, is_distributed=True)
                   grads_d = self.deep_optimzer.compute_gradients(self.deep_loss, var_list=tf.get_collection('deep'))
                else:
                   self.deep_optimzer = NPULossScaleOptimizer(self.deep_optimzer, loss_scale_manager2)
                   self.deep_optimzer = self.deep_optimzer.minimize(self.deep_loss,var_list=tf.get_collection("deep"))
                if rank_size > 1:
                    avg_grads_w = []
                    avg_grads_d = []
                    for grad, var in grads_w:
                        avg_grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                        avg_grads_w.append((avg_grad, var))
                    for grad, var in grads_d:
                        avg_grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                        avg_grads_d.append((avg_grad, var))
                    apply_gradient_op_w = self.wide_ptmzr.apply_gradients(avg_grads_w)
                    apply_gradient_op_d = self.deep_optimzer.apply_gradients(avg_grads_d)
                    self.train_op = tf.group(apply_gradient_op_d, apply_gradient_op_w)
                else:
                    self.train_op = tf.group(self.deep_optimzer, self.wide_ptmzr)



    def wide_forward(self, wide_wt, wide_id):
        wide_mask = tf.expand_dims(wide_wt, axis=2)
        gather = tf.gather(self.wide_w, wide_id)
        wide_part = tf.multiply(gather, wide_mask, name="wide_part")
        wide_output = tf.reshape((tf.reduce_sum(wide_part, axis=1) + self.wide_b), shape=[-1, ],
                                 name="wide_out")
        return wide_output

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
            # [batch, input_dim4lookup, embed_size]
            vx_embed = tf.multiply(tf.gather(self.embed_v, id_hldr), mask)

        hidden_output = tf.reshape(vx_embed, [-1, self.embed_dim])#*512
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(npu_ops.dropout(activate(act_func, hidden_output), keep_prob=keep_prob), self.h_w[i])
            else:
                hidden_output = tf.matmul(activate(act_func, hidden_output), self.h_w[i])
            hidden_output = hidden_output + self.h_b[i]

            if batch_norm:
                hidden_output = tf.layers.batch_normalization(
                    hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output
        return tf.reshape(hidden_output, [-1, ]), vx_embed

    def dump(self, model_path):
        var_map = {'W': self.fm_w.eval(),
                   'V': self.fm_v.eval(),
                   'b': self.fm_b.eval()}

        for i, (h_w_i, h_b_i) in enumerate(zip(self.h_w, self.h_b)):
            var_map['h%d_w' % (i+1)] = h_w_i.eval()
            var_map['h%d_b' % (i+1)] = h_b_i.eval()

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)


