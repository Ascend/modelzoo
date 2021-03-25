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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
implement the fc network with tensorflow
"""

from __future__ import absolute_import, division, print_function

from xt.model.tf_compat import K, tf, DTYPE_MAP

from xt.model.impala.default_config import ENTROPY_LOSS, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class ImpalaFCNet(XTModel):
    """docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_config.get("dtype", "float32"))

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]

        self.ph_state = None
        self.ph_adv = None
        self.out_actions = None
        self.out_val = None

        self.ph_target_action = None
        self.ph_target_val = None
        self.loss, self.optimizer, self.train_op = None, None, None
        self.saver = None

        super(ImpalaFCNet, self).__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = tf.placeholder(
            self.dtype, shape=(None, *self.state_dim,), name="state_input"
        )
        self.ph_adv = tf.placeholder(self.dtype, shape=(None, 1), name="adv")

        self.ph_target_action = tf.placeholder(
            self.dtype, shape=(None, self.action_dim), name="target_action"
        )
        self.ph_target_val = tf.placeholder(
            self.dtype, shape=(None, 1), name="target_value"
        )

        dense_layer = tf.layers.dense(
            inputs=self.ph_state, units=HIDDEN_SIZE, activation=tf.tanh
        )

        for _ in range(NUM_LAYERS - 1):
            dense_layer = tf.layers.dense(
                inputs=dense_layer, units=HIDDEN_SIZE, activation=tf.tanh
            )

        self.out_actions = tf.layers.dense(
            inputs=dense_layer, units=self.action_dim, activation=tf.nn.softmax
        )

        self.out_val = tf.layers.dense(inputs=dense_layer, units=1, activation=None)

        self.loss = 0.5 * tf.losses.mean_squared_error(
            self.ph_target_val, self.out_val
        ) + impala_loss(self.ph_adv, self.ph_target_action, self.out_actions)

        # self.optimizer = tf.train.AdamOptimizer(LR)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)
        return True

    def train(self, state, label):
        """train with sess.run"""
        with self.graph.as_default():
            _, loss = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.ph_state: state[0],
                    self.ph_adv: state[1],
                    self.ph_target_action: label[0],
                    self.ph_target_val: label[1],
                },
            )
        return loss

    def predict(self, state):
        """
        Do predict use the newest model.
        :param state:
        :return:
        """
        with self.graph.as_default():
            feed_dict = {self.ph_state: state[0]}
            return self.sess.run([self.out_actions, self.out_val], feed_dict)

    def save_model(self, file_name):
        ck_name = self.saver.save(
            self.sess, save_path=file_name, write_meta_graph=False
        )
        # print("save: ", ck_name)
        return ck_name

    def load_model(self, model_name, by_name=False):
        # print(">> load model: {}".format(model_name))
        self.saver.restore(self.sess, model_name)


def impala_loss(advantage, y_true, y_pred):
    """loss for impala"""
    policy = y_pred
    log_policy = K.log(policy + 1e-10)
    entropy = -policy * log_policy
    cross_entropy = -y_true * log_policy
    return K.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy)
