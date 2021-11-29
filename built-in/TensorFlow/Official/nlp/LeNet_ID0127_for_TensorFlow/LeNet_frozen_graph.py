#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================
"""LeNet frozen graph"""

import os
import time
import sys
import argparse

from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer


def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt


def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config


class LeNet(object):

    def __init__(self):
        pass

    def create(self, x):
        x = tf.reshape(x, [(- 1), 28, 28, 1])
        with tf.variable_scope('layer_1') as scope:
            w_1 = tf.get_variable('weights', shape=[5, 5, 1, 6])
            b_1 = tf.get_variable('bias', shape=[6])
        conv_1 = tf.nn.conv2d(x, w_1, strides=[1, 1, 1, 1], padding='SAME')
        act_1 = tf.sigmoid(tf.nn.bias_add(conv_1, b_1))
        max_pool_1 = tf.nn.max_pool(act_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('layer_2') as scope:
            w_2 = tf.get_variable('weights', shape=[5, 5, 6, 16])
            b_2 = tf.get_variable('bias', shape=[16])
        conv_2 = tf.nn.conv2d(max_pool_1, w_2, strides=[1, 1, 1, 1], padding='SAME')
        act_2 = tf.sigmoid(tf.nn.bias_add(conv_2, b_2))
        max_pool_2 = tf.nn.max_pool(act_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        flatten = tf.reshape(max_pool_2, shape=[(- 1), ((7 * 7) * 16)])
        with tf.variable_scope('fc_1') as scope:
            w_fc_1 = tf.get_variable('weight', shape=[((7 * 7) * 16), 120])
            b_fc_1 = tf.get_variable('bias', shape=[120], trainable=True)
        fc_1 = tf.nn.xw_plus_b(flatten, w_fc_1, b_fc_1)
        act_fc_1 = tf.nn.sigmoid(fc_1)
        with tf.variable_scope('fc_2') as scope:
            w_fc_2 = tf.get_variable('weight', shape=[120, 84])
            b_fc_2 = tf.get_variable('bias', shape=[84], trainable=True)
        fc_2 = tf.nn.xw_plus_b(act_fc_1, w_fc_2, b_fc_2)
        act_fc_2 = tf.nn.sigmoid(fc_2)
        with tf.variable_scope('fc_3') as scope:
            w_fc_3 = tf.get_variable('weight', shape=[84, 10])
            b_fc_3 = tf.get_variable('bias', shape=[10], trainable=True)
            tf.summary.histogram('weight', w_fc_3)
            tf.summary.histogram('bias', b_fc_3)
        fc_3 = tf.nn.xw_plus_b(act_fc_2, w_fc_3, b_fc_3)
        return fc_3


def frozen_graph():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(cur_dir, "test/output")
    if (not os.path.exists(output_path)):
        os.mkdir(output_path)

    output_node_names = ['Mean_1']

    with tf.Graph.as_default(), tf.Session(config=npu_session_config_init()) as sess:
        image = tf.placeholder(tf.float32, [None, 784], name='input_image')
        label = tf.placeholder(tf.float32, [None, 10], name='input_label')
        le_ins = LeNet()
        y_model = le_ins.create(image)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_model, labels=label))
        optimizer = npu_tf_optimizer(tf.train.AdamOptimizer())
        train_op = optimizer.minimize(loss)
        tf.summary.scalar('loss', loss)
        correct_pred = tf.equal(tf.argmax(y_model, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(output_path + '/logs')

        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(output_path + '/ckpt_npu')
        saver.restore(sess, model_file)
        tmp_g = sess.graph.as_graph_def()

        graph_varib_const = tf.graph_util.convert_variables_to_constants(sess, tmp_g, output_node_names)
        graph_frozen = tf.graph_util.remove_training_nodes(graph_varib_const)
        with tf.io.gfile.GFile('./LeNet_frozen_graph.pb', "wb") as file_write:
            file_write.write(graph_frozen.SerializeToString())


def main():
    frozen_graph()


if __name__ == '__main__':
    main()