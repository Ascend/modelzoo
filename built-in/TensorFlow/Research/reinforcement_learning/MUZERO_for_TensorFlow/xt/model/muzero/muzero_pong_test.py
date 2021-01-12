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
"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
import math
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers

from xt.model.tf_compat import K, Dense, MSE, Model, Conv2D, Input, Flatten, Lambda, Activation
from xt.model.muzero.default_config import LR, td_step, max_value
from xt.model.muzero.muzero_model import MuzeroBase, NetworkOutput, scale_gradient
from xt.model.muzero.muzero_atari import MuzeroAtari, residual_block

from xt.framework.register import Registers

@Registers.model.register
class MuzeroPongTest(MuzeroAtari):
    """docstring for ActorNetwork."""
    def create_model(self, model_info):
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)
        self.td_step = td_step
        self.max_value = max_value
        self.value_support_size = 21
        self.full_support_size = 2 * self.value_support_size + 2
        self.reward_support_size = 1
        self.reward_full_support_size = 2 * self.reward_support_size + 2

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
        obs_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(obs)

        # [convlayer = residual_block(convlayer, 256) for _ in range(16)]
        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(obs_1)
        convlayer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(256, activation='relu')(flattenlayer)
        # hidden = Lambda(hidden_normlize)(denselayer)
        hidden = denselayer
        return Model(inputs=obs, outputs=hidden)

    def create_policy_network(self):
        hidden_input = Input(shape=(256, ), name='hidden_input')
        hidden = Dense(128, activation='relu')(hidden_input)
        out_v = Dense(self.full_support_size, activation='softmax')(hidden)
        out_p = Dense(self.action_dim, activation='softmax')(hidden)
        # out_p = Dense(self.action_dim)(hidden)
        return Model(inputs=hidden_input, outputs=[out_p, out_v])

    def create_dyn_network(self):
        conditioned_hidden = Input(shape=(256 + self.action_dim, ))
        hidden = Dense(256, activation='relu')(conditioned_hidden)
        hidden = Dense(128, activation='relu')(hidden)
        out_h = Dense(256, activation='relu')(hidden)
        # out_h = Lambda(hidden_normlize)(out_h)
        out_r = Dense(self.reward_full_support_size, activation='softmax')(hidden)

        return Model(inputs=conditioned_hidden, outputs=[out_h, out_r])

    def initial_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            hidden = self.sess.run(self.out_rep, feed_dict)

            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)

            # print(value.shape)

            # feed_dict = {self.image: input_data}
            # init_infer_p, init_infer_v, init_infer_hidden = self.sess.run([self.init_infer_p, self.init_infer_v, self.init_infer_hidden], feed_dict)
            # print(policy, init_infer_p)
            # print(value, init_infer_v)
            # print(np.sum(hidden), np.sum(init_infer_hidden))
            value = self._value_transform(value[0], self.value_support_size)

        return NetworkOutput(value, 0, policy[0], hidden[0])

    def recurrent_inference(self, hidden_state, action):
        with self.graph.as_default():
            K.set_session(self.sess)
            action = np.eye(self.action_dim)[action]
            # action = action.reshape(6, 6, 1)
            action = np.expand_dims(action, 0)
            hidden_state = np.expand_dims(hidden_state, 0)
            # print("recu shape", hidden_state.shape, action.shape)
            conditioned_hidden = np.concatenate((hidden_state, action), axis=-1)
            feed_dict = {self.conditioned_hidden: conditioned_hidden}
            hidden, reward = self.sess.run([self.out_h, self.out_r], feed_dict)
            reward = self._value_transform(reward[0], self.reward_support_size)
            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)
            value = self._value_transform(value[0], self.value_support_size)
            # if math.isnan(value):
            # print("value", value, "reward", reward, np.sum(hidden), policy[0])
        return NetworkOutput(value, reward, policy[0], hidden[0])

    def value_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            hidden = self.sess.run(self.out_rep, feed_dict)

            feed_dict = {self.hidden: hidden}
            _, value = self.sess.run([self.out_p, self.out_v], feed_dict)

            value_list = []
            for value_data in value:
                value_list.append(self._value_transform(value_data, self.value_support_size))

        return value_list

    def build_graph(self):
        self.image = tf.placeholder(tf.float32, name="obs",
                shape=(None, ) + tuple(self.state_dim))
        self.action = tf.placeholder(tf.int32, name="action",
                shape=(None, self.td_step))
        self.target_value = tf.placeholder(tf.float32, name="value",
                shape=(None, ) + (1 + self.td_step, self.full_support_size))
        self.target_reward = tf.placeholder(tf.float32, name="reward",
                shape=(None, ) + (1 + self.td_step, self.reward_full_support_size))
        self.target_policy = tf.placeholder(tf.float32, name="policy",
                shape=(None, ) + (1 + self.td_step, self.action_dim))
        # self.target_policy = tf.placeholder(tf.float32, name="policy",
        #         shape=(None, self.action_dim))

        hidden_state = self.representation_network(self.image)
        policy_logits, value = self.policy_network(hidden_state)
        self.init_infer_p = policy_logits
        self.init_infer_v = value
        self.init_infer_hidden = hidden_state

        loss = 0
        # loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #         logits=value, labels=self.target_value[:,0]))
        # loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #             logits=policy_logits, labels=self.target_policy[:,0]))
        loss = cross_entropy(policy_logits, self.target_policy[:,0])
        loss += cross_entropy(value, self.target_value[:,0])
        # print(self.target_policy[:,0].shape, policy_logits.shape)

        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):
            action = tf.one_hot(self.action[:,i], self.action_dim)
            action = tf.reshape(action, (-1, self.action_dim,))
            conditioned_state = tf.concat((hidden_state, action), axis=-1)
            hidden_state, reward = self.dynamic_network(conditioned_state)
            policy_logits, value = self.policy_network(hidden_state)
            hidden_state = scale_gradient(hidden_state, 0.5)

            # l = MSE(reward, self.target_reward[:,i])
            # l = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #         logits=reward, labels=self.target_reward[:,i]))
            # l += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #         logits=value, labels=self.target_value[:,i+1]))
            # l += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #         logits=policy_logits, labels=self.target_policy[:,i+1]))
            l = cross_entropy(reward, self.target_reward[:, i])
            l += cross_entropy(policy_logits, self.target_policy[:, i+1])
            l += cross_entropy(value, self.target_value[:, i+1])
            loss += scale_gradient(l, gradient_scale)

        for weights in self.full_model.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
        self.loss = loss
        self.value = value
        return self.optimizer.minimize(loss)

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)

            target_value = self.conver_value(label[0], self.full_support_size, self.value_support_size)
            target_reward = self.conver_value(label[1], self.reward_full_support_size, self.reward_support_size)
            # print("target_value", label[0], target_value)
            # print("target_reward", label[1])
            # feed_dict = {self.image: state[0],
            #              self.action: state[1],
            #              self.target_value: target_value,
            #              self.target_reward: target_reward,
            #              self.target_policy: label[2]}
            # ret_value = self.sess.run([self.train_op, self.loss, self.value], feed_dict)

            feed_dict = {self.image: state[0],
                         self.action: state[1],
                         self.target_value: target_value,
                         self.target_reward: target_reward,
                         self.target_policy: label[2]}
            _, loss, value = self.sess.run([self.train_op, self.loss, self.init_infer_v], feed_dict)

            value_list = []
            for value_data in value:
                value_list.append(self._value_transform(value_data, self.value_support_size))
            # print("loss", np.mean(ret_value[1]), ret_value[2])
            # print(self.target_value.shape, np.sum(target_value), target_value.shape)
            # print("loss", np.mean(ret_value[1]))
            return np.mean(loss), value_list


    def conver_value(self, target_value, full_size, support_size):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros((target_value.shape[0:2] + (full_size,)))
        # print(target_value.shape[0:2])
        target_value = np.clip(target_value, -support_size, support_size)
        batch_size = targets.shape[0]
        td_size = targets.shape[1]
        for i in range(batch_size):
            # sqrt_value = np.sign(target_value[i]) * (np.sqrt(np.abs(target_value[i]) + 1) - 1) + 0.001 * target_value[i]
            floor_value = np.floor(target_value[i]).astype(int)
            # rest = sqrt_value - floor_value
            rest = target_value[i] - floor_value
            index =  floor_value.astype(int) + support_size
            try:
                targets[i, range(td_size), index] = 1 - rest
                targets[i, range(td_size), index + 1] = rest
            except:
                print(target_value[i])
                raise NotImplementedError

        return targets

    def _value_transform(self, value_support, support_size):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        # probs = self._softmax(value_support)
        probs = value_support
        value = np.dot(probs, range(-support_size, support_size+2))
        # value = np.sign(value) * (
        #     ((np.sqrt(1 + 4 * 0.001 * (np.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001))
        #     ** 2
        #     - 1
        # )
        value_clip = np.clip(value, -support_size, support_size)
        # if math.isnan(value_clip):
        #     print("nan vlaue", value_support, probs, value)
        return np.asscalar(value_clip)

def hidden_normlize(hidden):
    hidden_max = tf.reduce_max(hidden, axis=-1, keepdims=True)
    hidden_min = tf.reduce_min(hidden, axis=-1, keepdims=True)
    hidden_norm = (hidden - hidden_min) / (hidden_max - hidden_min + 1e-10)
    return hidden_norm

def cross_entropy(pred_p, target_p):
    return tf.reduce_mean(-target_p * tf.log(pred_p + 1e-10))
