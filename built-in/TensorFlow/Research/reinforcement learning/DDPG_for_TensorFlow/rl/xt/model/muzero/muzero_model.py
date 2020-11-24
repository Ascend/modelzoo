"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
import typing
from typing import List
import math
import numpy as np

import tensorflow as tf
from xt.model.tf_compat import K, Dense, MSE, Model, Sequential

from xt.model import XTModel
from xt.model.muzero.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LR, td_step, max_value
from xt.util.common import import_config

from xt.framework.register import Registers

# pylint: disable=W0201
@Registers.model.register
class MuzeroModel(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(MuzeroModel, self).__init__(model_info)


    def create_model(self, model_info):
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)
        self.td_step = td_step
        self.max_value = max_value
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        representation_network = Sequential([Dense(HIDDEN1_UNITS, activation='relu', dtype='float32', name="rep1"),
                                             Dense(HIDDEN2_UNITS, activation='tanh', dtype='float32', name="rep2")])
        value_network = Sequential([Dense(HIDDEN1_UNITS, activation='relu', dtype='float32'),
                                    Dense((self.value_support_size * 2) + 1, dtype='float32')])
        policy_network = Sequential([Dense(HIDDEN1_UNITS, activation='relu', dtype='float32'),
                                     Dense(self.action_dim, dtype='float32')
                                     ])
        dynamic_network = Sequential([Dense(HIDDEN1_UNITS, activation='relu', dtype='float32'),
                                      Dense(HIDDEN2_UNITS, activation='tanh', dtype='float32')])
        reward_network = Sequential([Dense(16, activation='relu', dtype='float32'),
                                     Dense(1, dtype='float32')])
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.save_all_model = SaveModel(representation_network, dynamic_network,
                                        reward_network, value_network, policy_network)
        # print(dynamic_network)
        self.initial_model = InitialModel(representation_network, value_network, policy_network)
        self.recurrent_model = RecurrentModel(dynamic_network, reward_network,
                                              value_network, policy_network)


        if model_info.get("type"):
            self.train_op = self.build_graph()
            self.sess.run(tf.initialize_all_variables())
        # self.recurrent_model.summary()
        return self.recurrent_model

    def cb_get_variables(self):
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables


    def initial_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)
        # input = np.expand_dims(input, 0)
            hidden, value, policy = self.initial_model.predict(input_data)
            value = self._value_transform(value)

        return NetworkOutput(value, 0, policy, hidden)

    def recurrent_inference(self, hidden_state, action):
        with self.graph.as_default():
            K.set_session(self.sess)
            action = np.expand_dims(np.eye(self.action_dim)[action], 0)
            conditioned_hidden = np.hstack((hidden_state, action))
            hidden, reward, value, policy = self.recurrent_model.predict(conditioned_hidden)
            value = self._value_transform(value)

        return NetworkOutput(value, reward, policy, hidden)

    def build_graph(self):
        self.image = tf.placeholder(tf.float32, name="obs",
                                    shape=(None, ) + tuple(self.state_dim))
        self.action = tf.placeholder(tf.int32, name="action",
                                     shape=(None, self.td_step))
        target_value_shape = (None, ) + (1 + self.td_step, (self.value_support_size * 2) + 1)
        self.target_value = tf.placeholder(tf.float32, name="value",
                                           shape=target_value_shape)
        self.target_reward = tf.placeholder(tf.float32, name="reward",
                                            shape=(None, ) + (1 + self.td_step, ))
        self.target_policy = tf.placeholder(tf.float32, name="policy",
                                            shape=(None, ) + (1 + self.td_step, self.action_dim))
        hidden_state, value, policy_logits = self.initial_model(self.image)

        loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=value, labels=self.target_value[:, 0]))
        loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=self.target_policy[:, 0]))

        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):

            action = tf.one_hot(self.action[:, i], self.action_dim)

            # print("hidden_state.shape", hidden_state, action)
            conditioned_state = tf.concat((hidden_state, action), axis=1)
            hidden_state, reward, value, policy_logits = self.recurrent_model(conditioned_state)

            hidden_state = scale_gradient(hidden_state, 0.5)

            tmp_loss = MSE(reward, self.target_reward[:, i+1])
            tmp_loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=value, labels=self.target_value[:, i+1]))
            tmp_loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=self.target_policy[:, i+1]))
            loss += scale_gradient(tmp_loss, gradient_scale)

        for weights in self.save_all_model.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
        self.loss = loss
        return self.optimizer.minimize(loss)



    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)

            target_value = self.conver_value(label[0])
            feed_dict = {self.image: state[0],
                         self.action: state[1],
                         self.target_value: target_value,
                         self.target_reward: label[1],
                         self.target_policy: label[2]}
            ret_value = self.sess.run([self.train_op, self.loss], feed_dict)

            #print("loss", np.mean(ret_value[1]))
            return np.mean(ret_value[1])

    def save_model(self, file_name):
        """save weights into .h5 file"""
        with self.graph.as_default():
            K.set_session(self.sess)

            self.initial_model.save_weights(file_name + "init" + ".h5", overwrite=True)
            self.recurrent_model.save_weights(file_name + "recur" + ".h5", overwrite=True)

        return file_name

    def load_model(self, model_name, by_name=False):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.initial_model.load_weights(model_name + "init" + ".h5", by_name=False)
            self.recurrent_model.load_weights(model_name + "recur" + ".h5", by_name=False)


    def conver_value(self, target_value):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros((target_value.shape + ((self.value_support_size * 2) + 1,)))
        batch_size = target_value.shape[0]
        td_size = target_value.shape[1]
        for i in range(batch_size):
            sqrt_value = np.sign(target_value[i]) * np.sqrt(np.abs(target_value[i]))
            floor_value = np.floor(sqrt_value).astype(int)
            rest = sqrt_value - floor_value
            index = floor_value.astype(int) + self.value_support_size
            targets[i, range(td_size), index] = 1 - rest
            targets[i, range(td_size), index + 1] = rest

        return targets


    def _value_transform(self, value_support):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(-self.value_support_size, self.value_support_size+1))
        value = np.sign(value) * (np.asscalar(value) ** 2)
        return np.asscalar(value)

    @staticmethod
    def _softmax(values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: List[int]
    hidden_state: List[float]

class InitialModel(Model):
    """Model that combine the representation and prediction (value+policy) network."""
    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, image):
        hidden_representation = self.representation_network(image)

        value = self.value_network(hidden_representation)

        policy_logits = self.policy_network(hidden_representation)

        return hidden_representation, value, policy_logits

class RecurrentModel(Model):
    """Model that combine the dynamic, reward and prediction (value+policy) network."""

    def __init__(self, dynamic_network: Model, reward_network: Model, value_network: Model, policy_network: Model):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)

        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits

class SaveModel(Model):
    """Model that combine the dynamic, reward and prediction (value+policy) network."""

    def __init__(self, representation_network: Model, dynamic_network: Model, reward_network: Model,
                 value_network: Model, policy_network: Model):
        super(SaveModel, self).__init__()
        self.representation_network = representation_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)

        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)

        return hidden_representation, reward, value, policy_logits
