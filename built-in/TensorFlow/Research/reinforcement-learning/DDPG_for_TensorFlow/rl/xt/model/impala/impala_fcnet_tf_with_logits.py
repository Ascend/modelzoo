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

import xt.model.impala.vtrace_tf as vtrace
from xt.framework.register import Registers
from xt.model import XTModel
from xt.model.impala.default_config import ENTROPY_LOSS, GAMMA, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model.tf_compat import DTYPE_MAP, Adam, Dense, Input, K, Model, tf
from xt.util.common import import_config


@Registers.model.register
class ImpalaFCNetV2(XTModel):
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
        self.policy_logits, self.baseline = None, None

        self.ph_behavior_logits = None
        self.ph_actions = None
        self.ph_dones = None
        self.ph_rewards = None
        self.loss, self.optimizer, self.train_op = None, None, None
        self.grad_norm_clip = 40.0
        self.sample_batch_steps = 50

        self.saver = None

        super(ImpalaFCNetV2, self).__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = tf.placeholder(self.dtype, shape=(
            None,
            *self.state_dim,
        ), name="state_input")
        dense_layer = tf.layers.dense(inputs=self.ph_state, units=HIDDEN_SIZE, activation=tf.tanh)

        for _ in range(NUM_LAYERS - 1):
            dense_layer = tf.layers.dense(inputs=dense_layer, units=HIDDEN_SIZE, activation=tf.tanh)

        self.policy_logits = tf.layers.dense(inputs=dense_layer, units=self.action_dim, activation=None)
        print("self.policy_logits: ", self.policy_logits)
        # self.baseline = tf.layers.dense(inputs=dense_layer, units=1, activation=None)
        self.baseline = tf.squeeze(tf.layers.dense(inputs=dense_layer, units=1, activation=None), 1)

        # self.out_actions = tf.layers.dense(
        #     inputs=dense_layer, units=self.action_dim, activation=tf.nn.softmax
        # )
        print("self.baseline: ", self.baseline)
        self.out_actions = tf.squeeze(
            tf.multinomial(self.policy_logits, num_samples=1, output_dtype=tf.int32),
            1,
            name="out_action",
        )
        # self.out_actions = tf.multinomial(self.policy_logits,
        #                                   num_samples=1,
        #                                   output_dtype=tf.int32,
        #                                   name="out_action",
        # )
        # print("self.out_actions: ", self.out_actions)

        # create learner
        self.ph_behavior_logits = tf.placeholder(self.dtype, shape=(None, self.action_dim), name="ph_behavior_logits")

        # self.ph_actions = tf.placeholder(tf.int32, shape=(None, 1), name="ph_action")
        self.ph_actions = tf.placeholder(tf.int32, shape=(None, ), name="ph_action")
        self.ph_dones = tf.placeholder(tf.bool, shape=(None, ), name="ph_dones")
        self.ph_rewards = tf.placeholder(self.dtype, shape=(None, ), name="ph_rewards")

        # discounts = tf.to_float(~self.ph_dones) * GAMMA
        # clipped_rewards = tf.clip_by_value(self.ph_rewards, -1, 1)

        # loss
        """
        Split the tensor into batches at known episode cut boundaries. 
        [B * T] -> [T, B]
        """

        T = self.sample_batch_steps

        def split_batches(tensor, drop_last=False):
            B = tf.shape(tensor)[0] // T
            print("raw: ", tensor)
            rs = tf.reshape(tensor, tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))
            print(rs)

            # swap B and T axes
            res = tf.transpose(rs, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

            if drop_last:
                return res[:-1]
            return res

        self.loss = vtrace_loss(
            behavior_policy_logits=split_batches(self.ph_behavior_logits, drop_last=True),
            target_policy_logits=split_batches(self.policy_logits, drop_last=True),
            actions=split_batches(self.ph_actions, drop_last=True),
            discounts=split_batches(tf.cast(~self.ph_dones, tf.float32) * GAMMA, drop_last=True),
            rewards=split_batches(tf.clip_by_value(self.ph_rewards, -1, 1), drop_last=True),
            # rewards=tf.clip_by_value(self.ph_rewards, -1, 1),
            values=split_batches(self.baseline, drop_last=True),
            bootstrap_value=split_batches(self.baseline)[-1],
        )

        # self.optimizer = tf.train.AdamOptimizer(LR)
        opt_type = "adam"
        if opt_type == "adam":
            optimizer = tf.train.AdamOptimizer(LR)
            # self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)

        # Optimise
        elif opt_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(LR, decay=0.99, epsilon=0.1, centered=True)
        else:
            raise KeyError("invalid opt_type: {}".format(opt_type))

        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(
            grad if grad is None else tf.clip_by_norm(grad, clip_norm=self.grad_norm_clip),
            var,
        ) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        return True

    def train(self, state, label):
        """train with sess.run"""

        behavior_logits, actions, dones, rewards = label
        with self.graph.as_default():
            _, loss = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.ph_state: state,
                    self.ph_behavior_logits: behavior_logits,
                    self.ph_actions: actions,
                    self.ph_dones: dones,
                    self.ph_rewards: rewards,
                },
            )
        return loss

    def predict(self, state):
        """
        action_logits, action_val, value
        Do predict use the newest model.
        :param state:
        :return:
        """
        with self.graph.as_default():
            feed_dict = {self.ph_state: state}
            return self.sess.run([self.policy_logits, self.baseline, self.out_actions], feed_dict)

    def save_model(self, file_name):
        ck_name = self.saver.save(self.sess, save_path=file_name, write_meta_graph=False)
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


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return 0.5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)


def vtrace_loss(
        behavior_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
):

    # clip reward
    # clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    # discounts = tf.to_float(~dones) * FLAGS.discounting
    with tf.device("/cpu"):
        vtrace_returns = vtrace.from_logits(
            behaviour_policy_logits=behavior_policy_logits,
            target_policy_logits=target_policy_logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
        )

    total_loss = compute_policy_gradient_loss(target_policy_logits, actions, vtrace_returns.pg_advantages)
    total_loss += 0.5 * compute_baseline_loss(vtrace_returns.vs - values)
    total_loss += 0.01 * compute_entropy_loss(target_policy_logits)

    return total_loss
