"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
import typing
from typing import List
import math
import numpy as np

import tensorflow as tf
from xt.model.tf_compat import K, Dense, MSE, Model, Sequential, Input, Lambda

from xt.model import XTModel
from xt.model.muzero.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LR, td_step, max_value
from xt.util.common import import_config

from xt.framework.register import Registers

# pylint: disable=W0201
@Registers.model
class MuzeroModel(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(MuzeroModel, self).__init__(model_info)


    def create_model(self, model_info):
        self.td_step = td_step
        self.max_value = max_value
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        self.full_support_size = 2 * self.value_support_size + 1
        # self.full_support_size = self.value_support_size
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)

        self.representation_network = self.create_rep_network()
        self.policy_network = self.create_policy_network()
        self.dynamic_network = self.create_dyn_network()

        self.out_v = self.policy_network.outputs[1]
        self.out_p = self.policy_network.outputs[0]
        self.out_h = self.dynamic_network.outputs[0]
        self.out_r = self.dynamic_network.outputs[1]
        self.out_rep= self.representation_network.outputs[0]
        self.hidden = self.policy_network.inputs[0]
        self.conditioned_hidden = self.dynamic_network.inputs[0]
        self.obs = self.representation_network.inputs[0]
        self.full_model = MuzeroBase(self.representation_network,
                                self.dynamic_network,
                                self.policy_network)

        self.train_op = self.build_graph()
        self.sess.run(tf.initialize_all_variables())

        return self.full_model

    def create_rep_network(self):
        obs = Input(shape=self.state_dim, name='rep_input')
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(obs)
        out_rep = Dense(HIDDEN2_UNITS, activation='relu')(hidden)
        out_rep = Lambda(hidden_normlize)(out_rep)
        return Model(inputs=obs, outputs=out_rep)

    def create_policy_network(self):
        hidden_input = Input(shape=HIDDEN2_UNITS, name='hidden_input')
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(hidden_input)
        # hidden1 = Dense(HIDDEN1_UNITS, activation='relu')(hidden_input)
        out_v = Dense(self.full_support_size)(hidden)
        out_p = Dense(self.action_dim)(hidden)
        return Model(inputs=hidden_input, outputs=[out_p, out_v])

    def create_dyn_network(self):
        conditioned_hidden = Input(shape=HIDDEN2_UNITS + self.action_dim)
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(conditioned_hidden)
        out_h = Dense(HIDDEN2_UNITS, activation='relu')(hidden)
        out_h = Lambda(hidden_normlize)(out_h)
        # hidden1 = Dense(16, activation='relu')(conditioned_hidden)
        out_r = Dense(1)(hidden)
        return Model(inputs=conditioned_hidden, outputs=[out_h, out_r])

    def initial_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            hidden = self.sess.run(self.out_rep, feed_dict)

            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)
            value = self._value_transform(value[0])

        return NetworkOutput(value, 0, policy[0], hidden[0])

    def recurrent_inference(self, hidden_state, action):
        with self.graph.as_default():
            K.set_session(self.sess)
            action = np.expand_dims(np.eye(self.action_dim)[action], 0)
            hidden_state = np.expand_dims(hidden_state, 0)
            conditioned_hidden = np.hstack((hidden_state, action))
            feed_dict = {self.conditioned_hidden: conditioned_hidden}
            hidden, reward = self.sess.run([self.out_h, self.out_r], feed_dict)

            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)
            value = self._value_transform(value[0])
            # print("value", value, reward, hidden)

        return NetworkOutput(value, reward[0], policy[0], hidden[0])

    def build_graph(self):
        self.image = tf.placeholder(tf.float32, name="obs",
                                    shape=(None, ) + tuple(self.state_dim))
        self.action = tf.placeholder(tf.int32, name="action",
                                     shape=(None, self.td_step))
        target_value_shape = (None, ) + (1 + self.td_step, self.full_support_size)
        self.target_value = tf.placeholder(tf.float32, name="value",
                                           shape=target_value_shape)
        self.target_reward = tf.placeholder(tf.float32, name="reward",
                                            shape=(None, ) + (1 + self.td_step, ))
        self.target_policy = tf.placeholder(tf.float32, name="policy",
                                            shape=(None, ) + (1 + self.td_step, self.action_dim))

        hidden_state = self.representation_network(self.image)
        policy_logits, value = self.policy_network(hidden_state)

        loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=value, labels=self.target_value[:, 0]))
        loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=self.target_policy[:, 0]))

        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):
            action = tf.one_hot(self.action[:, i], self.action_dim)

            conditioned_state = tf.concat((hidden_state, action), axis=1)
            hidden_state, reward = self.dynamic_network(conditioned_state)
            policy_logits, value = self.policy_network(hidden_state)
            hidden_state = scale_gradient(hidden_state, 0.5)

            tmp_loss = MSE(reward, self.target_reward[:, i])
            tmp_loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=value, labels=self.target_value[:, i+1]))
            tmp_loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=self.target_policy[:, i+1]))
            loss += scale_gradient(tmp_loss, gradient_scale)

        for weights in self.full_model.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
        self.loss = loss
        return self.optimizer.minimize(loss)

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)

            target_value = self.conver_value(label[0])
            # print("target_value", label[0])
            # print("target_reward", label[1])
            feed_dict = {self.image: state[0],
                         self.action: state[1],
                         self.target_value: target_value,
                         self.target_reward: label[1],
                         self.target_policy: label[2]}
            ret_value = self.sess.run([self.train_op, self.loss], feed_dict)

            return np.mean(ret_value[1])

    def conver_value(self, target_value):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros(target_value.shape + (self.full_support_size, ))
        batch_size = target_value.shape[0]
        td_size = target_value.shape[1]

        for i in range(batch_size):
            sqrt_value = np.sqrt(np.abs(target_value[i])) * np.sign(target_value[i])
            floor_value = np.floor(sqrt_value).astype(int)
            rest = sqrt_value - floor_value

            index =  floor_value.astype(int) + self.value_support_size
            # index =  floor_value.astype(int)
            targets[i, range(td_size), index] = 1 - rest
            targets[i, range(td_size), index + 1] = rest

        return targets

    def _value_transform(self, value_support):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = self._softmax(value_support)
        value = np.dot(value, range(-self.value_support_size, self.value_support_size + 1))
        # value = np.dot(value, range(0, self.value_support_size))
        value = np.sign(value) * (np.asscalar(value) ** 2)
        # value = np.clip(value, 0, 200)
        return np.asscalar(value)

    def value_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            hidden = self.sess.run(self.out_rep, feed_dict)

            feed_dict = {self.hidden: hidden}
            _, value = self.sess.run([self.out_p, self.out_v], feed_dict)

            value_list = []
            for value_data in value:
                value_list.append(self._value_transform(value_data))

        return value_list

    @staticmethod
    def _softmax(values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

def hidden_normlize(hidden):
    hidden_max = tf.reduce_max(hidden, axis=-1, keepdims=True)
    hidden_min = tf.reduce_min(hidden, axis=-1, keepdims=True)
    hidden_norm = (hidden - hidden_min) / (hidden_max - hidden_min + 1e-10)
    return hidden_norm

class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: List[int]
    hidden_state: List[float]

class MuzeroBase(Model):
    """Model that combine the representation and prediction (value+policy) network."""
    def __init__(self, representation_network: Model, dynamic_network: Model, policy_network: Model):
        super().__init__()
        self.representation_network = representation_network
        self.dynamic_network = dynamic_network
        self.policy_network = policy_network
