#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
implement the cnn network with tensorflow
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import xt.model.impala.vtrace_tf as vtrace
from tensorflow.python.util import deprecation
from xt.framework.register import Registers
from xt.model import XTModel
from xt.model.impala.default_config import GAMMA, LR
from xt.model.tf_compat import (
    DTYPE_MAP,
    AdamOptimizer,
    Conv2D,
    Flatten,
    Lambda,
    Saver,
    global_variables_initializer,
    piecewise_constant,
    tf,
)
from xt.model.atari_model import get_atari_filter
from xt.model.tf_utils import TFVariables, restore_tf_variable
from xt.util.common import import_config

deprecation._PRINT_DEPRECATION_WARNINGS = False


@Registers.model
class ImpalaCNNNetV2(XTModel):
    """docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_config.get("dtype", "float32"))

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]
        self.filter_arch = get_atari_filter(self.state_dim)

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
        self.explore_paras = None
        self.actor_var = None  # store weights for agent

        super(ImpalaCNNNetV2, self).__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = tf.placeholder(
            tf.int8, shape=(None, *self.state_dim,), name="state_input"
        )

        with tf.variable_scope("explore_agent"):
            state_input = Lambda(lambda x: tf.cast(x, dtype="float32") / 128.0)(
                self.ph_state
            )
            last_layer = state_input

            for (out_size, kernel, stride) in self.filter_arch[:-1]:
                last_layer = Conv2D(out_size, (kernel, kernel),
                                    strides=(stride, stride), activation="relu",
                                    padding="same")(last_layer)

            # last convolution
            (out_size, kernel, stride) = self.filter_arch[-1]
            convolution_layer = Conv2D(out_size, (kernel, kernel),
                                       strides=(stride, stride), activation="relu",
                                       padding="valid")(last_layer)

            self.policy_logits = tf.squeeze(
                Conv2D(self.action_dim, (1, 1), padding="same")(convolution_layer),
                axis=[1, 2])

            baseline_flat = Flatten()(convolution_layer)
            self.baseline = tf.squeeze(
                tf.layers.dense(
                    inputs=baseline_flat,
                    units=1,
                    activation=None,
                    kernel_initializer=norm_initializer(0.01),
                ),
                1,
            )
            self.out_actions = tf.squeeze(
                tf.multinomial(
                    self.policy_logits, num_samples=1, output_dtype=tf.int32
                ),
                1,
                name="out_action",
            )

        # create learner
        self.ph_behavior_logits = tf.placeholder(
            self.dtype, shape=(None, self.action_dim), name="ph_behavior_logits"
        )

        self.ph_actions = tf.placeholder(tf.int32, shape=(None,), name="ph_action")
        self.ph_dones = tf.placeholder(tf.bool, shape=(None,), name="ph_dones")
        self.ph_rewards = tf.placeholder(self.dtype, shape=(None,), name="ph_rewards")

        # Split the tensor into batches at known episode cut boundaries.
        # [batch_count * batch_step] -> [batch_step, batch_count]
        batch_step = self.sample_batch_steps

        def split_batches(tensor, drop_last=False):
            batch_count = tf.shape(tensor)[0] // batch_step
            reshape_tensor = tf.reshape(
                tensor, tf.concat([[batch_count, batch_step], tf.shape(tensor)[1:]], axis=0))

            # swap B and T axes
            res = tf.transpose(
                reshape_tensor, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0])))
            )

            if drop_last:
                return res[:-1]
            return res

        self.loss = vtrace_loss(
            behavior_policy_logits=split_batches(
                self.ph_behavior_logits, drop_last=True
            ),
            target_policy_logits=split_batches(self.policy_logits, drop_last=True),
            actions=split_batches(self.ph_actions, drop_last=True),
            discounts=split_batches(
                tf.cast(~self.ph_dones, tf.float32) * GAMMA, drop_last=True
            ),
            rewards=split_batches(
                tf.clip_by_value(self.ph_rewards, -1, 1), drop_last=True
            ),
            values=split_batches(self.baseline, drop_last=True),
            bootstrap_value=split_batches(self.baseline)[-1],
        )

        opt_type = "adam"
        learning_rate, global_step = self._get_lr()
        if opt_type == "adam":
            optimizer = AdamOptimizer(LR)
            # optimizer = AdamOptimizer(learning_rate)
        elif opt_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                LR, decay=0.99, epsilon=0.1, centered=True
            )
        else:
            raise KeyError("invalid opt_type: {}".format(opt_type))

        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [
            (grad if grad is None else tf.clip_by_norm(
                grad, clip_norm=self.grad_norm_clip),
             var) for grad, var in grads_and_vars
        ]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        self.actor_var = TFVariables(self.out_actions, self.sess)

        self.sess.run(global_variables_initializer())
        # self.saver = Saver(max_to_keep=100)

        self.explore_paras = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="explore_agent"
        )

        self.saver = Saver({t.name: t for t in self.explore_paras}, max_to_keep=100)

        return True

    @staticmethod
    def _get_lr(values=None, boundaries=None):
        """make dynamic learning rate"""
        values = [0.0025, 0.002, 0.001]
        boundaries = np.array([20000 / 1000, 200000 / 1000]).astype(np.int32).tolist()
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        learning_rate = piecewise_constant(global_step, boundaries, values)
        return learning_rate, global_step

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
            return self.sess.run(
                [self.policy_logits, self.baseline, self.out_actions], feed_dict
            )

    def save_model(self, file_name):
        """save model without meta graph"""
        ck_name = self.saver.save(
            self.sess, save_path=file_name, write_meta_graph=False
        )
        return ck_name

    def load_model(self, model_name, by_name=False):
        """load model with inference variables."""
        # print(">> load model: {}".format(model_name))
        # self.saver.restore(self.sess, model_name)
        restore_tf_variable(self.sess, self.explore_paras, model_name)

    def set_weights(self, weights):
        """set weight with memory tensor"""
        with self.graph.as_default():
            self.actor_var.set_weights(weights)

    def get_weights(self):
        """get weights"""
        # print("model get weight")
        with self.graph.as_default():
            return self.actor_var.get_weights()


def compute_baseline_loss(advantages):
    """Loss for the baseline, summed over the time dimension.
    Multiply by 0.5 to match the standard update rule:"""
    # d(loss) / d(baseline) = advantage
    return 0.5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    """calculate entropy loss"""
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    """calculate policy gradient loss"""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits
    )
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
    """vtrace loss from impala algorithm."""
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

    total_loss = compute_policy_gradient_loss(
        target_policy_logits, actions, vtrace_returns.pg_advantages
    )
    total_loss += 0.5 * compute_baseline_loss(vtrace_returns.vs - values)
    total_loss += 0.01 * compute_entropy_loss(target_policy_logits)

    return total_loss


def norm_initializer(std=0.5):
    """custom norm initializer for op"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer
