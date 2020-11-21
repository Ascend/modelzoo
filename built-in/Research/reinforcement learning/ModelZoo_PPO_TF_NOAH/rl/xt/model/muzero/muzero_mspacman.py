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
from xt.model.muzero.muzero_pong_test import MuzeroPongTest

from xt.framework.register import Registers

@Registers.model
class MuzeroMsPacman(MuzeroPongTest):
    """docstring for ActorNetwork."""
    def create_model(self, model_info):
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)
        self.td_step = td_step
        self.max_value = max_value
        self.value_support_size = 100
        self.full_support_size = self.value_support_size + 2
        self.value_clip = 8546
        self.reward_support_size = 30
        self.reward_full_support_size = self.reward_support_size + 2
        self.reward_clip = 904

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
        self.infer_count = 0
        self.load_flg = False
        self.init_policy = np.array([1/self.action_dim] * self.action_dim)
        self.init_hidden = np.array([1] * 256)

        return self.full_model

    # def create_rep_network(self):
    #     obs = Input(shape=self.state_dim, name='rep_input')
    #     obs_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(obs)
    #
    #     # [convlayer = residual_block(convlayer, 256) for _ in range(16)]
    #     convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(obs_1)
    #     convlayer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
    #     convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
    #     flattenlayer = Flatten()(convlayer)
    #     denselayer = Dense(256, activation='relu')(flattenlayer)
    #     # hidden = Lambda(hidden_normlize)(denselayer)
    #     hidden = denselayer
    #     return Model(inputs=obs, outputs=hidden)
    #
    # def create_policy_network(self):
    #     hidden_input = Input(shape=(256, ), name='hidden_input')
    #     hidden = Dense(128, activation='relu')(hidden_input)
    #     out_v = Dense(self.full_support_size)(hidden)
    #     out_p = Dense(self.action_dim)(hidden)
    #     return Model(inputs=hidden_input, outputs=[out_p, out_v])
    #
    # def create_dyn_network(self):
    #     conditioned_hidden = Input(shape=(256 + self.action_dim, ))
    #     hidden = Dense(256, activation='relu')(conditioned_hidden)
    #     hidden = Dense(128, activation='relu')(hidden)
    #     out_h = Dense(256, activation='relu')(hidden)
    #     out_h = Lambda(hidden_normlize)(out_h)
    #     # out_h = layers.BatchNormalization()(out_h)
    #
    #     out_r = Dense(self.reward_full_support_size)(hidden)
    #
    #     return Model(inputs=conditioned_hidden, outputs=[out_h, out_r])

    def initial_inference(self, input_data):
        if self.load_flg == False:
            return NetworkOutput(0, 0, self.init_policy, self.init_hidden)

        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            hidden = self.sess.run(self.out_rep, feed_dict)

            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)

            # print(value.shape)
            value = self._value_transform(value[0], self.value_support_size, self.value_clip)

        return NetworkOutput(value, 0, policy[0], hidden[0])

    def recurrent_inference(self, hidden_state, action):
        if self.load_flg == False:
            return NetworkOutput(0, 0, self.init_policy, self.init_hidden)

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
            reward = self._value_transform(reward[0], self.reward_support_size, self.reward_clip)
            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)
            value = self._value_transform(value[0], self.value_support_size, self.value_clip)
            # if math.isnan(value):
            self.infer_count += 1
            # if self.infer_count % 1000 == 0:
            #     print("value", value, "reward", reward, np.sum(hidden))
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
                value_list.append(self._value_transform(value_data, self.value_support_size, self.value_clip))

        return value_list

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)

            target_value = self.conver_value(label[0], self.full_support_size,
                                             self.value_support_size, self.value_clip)
            target_reward = self.conver_value(label[1], self.reward_full_support_size,
                                              self.reward_support_size, self.reward_clip)
            # print("target_value", label[0], target_value)
            # print("target_value", label[0])
            # print("target_reward", label[1])
            try:

                feed_dict = {self.image: state[0],
                             self.action: state[1],
                             self.loss_weights: state[2],
                             self.target_value: target_value,
                             self.target_reward: target_reward,
                             self.target_policy: label[2]}
                ret_value = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            except Exception as e:
                print("input type", state[1])
                print("input type", state[2])
                print("input type", label[1])
                print("input type", state[0])
                print("input type", target_value)
                print("input type", target_reward)
                raise e
            # print("loss", np.mean(ret_value[1]), ret_value[2])
            # print("loss", np.mean(ret_value[1]))
            return np.mean(ret_value[1])

    def conver_value(self, target_value, full_size, support_size, clip):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros((target_value.shape + (full_size,)))
        # print(target_value)
        target_value = np.clip(target_value, 0, clip)
        batch_size = targets.shape[0]
        td_size = targets.shape[1]
        for i in range(batch_size):
            sqrt_value = np.sign(target_value[i]) * (np.sqrt(np.abs(target_value[i]) + 1) - 1) + 0.001 * target_value[i]
            floor_value = np.floor(sqrt_value).astype(int)
            rest = sqrt_value - floor_value
            # rest = target_value[i] - floor_value
            index =  floor_value.astype(int)
            try:
                targets[i, range(td_size), index] = 1 - rest
                targets[i, range(td_size), index + 1] = rest
            except:
                print(target_value[i])
                raise NotImplementedError

        return targets

    def _value_transform(self, value_support, support_size, clip):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        # probs = self._softmax(value_support)
        probs = value_support
        value = np.dot(probs, range(0, support_size+2))
        value = np.sign(value) * (
            ((np.sqrt(1 + 4 * 0.001 * (np.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001))
            ** 2
            - 1
        )
        value_clip = np.clip(value, 0, clip)
        # if math.isnan(value_clip):
        #     print("nan vlaue", value_support, probs, value)
        return np.asscalar(value_clip)

    def load_model(self, model_name, by_name=False):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.load_weights(model_name, by_name)

        self.load_flg = True

def hidden_normlize(hidden):
    hidden_max = tf.reduce_max(hidden, axis=-1, keepdims=True)
    hidden_min = tf.reduce_min(hidden, axis=-1, keepdims=True)
    hidden_norm = (hidden - hidden_min) / (hidden_max - hidden_min + 1e-10)
    return hidden_norm
