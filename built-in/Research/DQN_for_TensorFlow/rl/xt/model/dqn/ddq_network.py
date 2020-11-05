"""
@Author: Huaqiang Wang,
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

import tensorflow as tf

from xt.model.tf_compat import Dense, Input, Lambda, Model, Adam
from xt.model.dqn.default_config import LEARNING_RATE
from xt.model import XTModel

from xt.framework.register import Registers


@Registers.model.register
class DdqNetwork(XTModel):
    """QNetwork for DQN with batch normalization
    """
    def __init__(self, model_info):
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(DdqNetwork, self).__init__(model_info)

    def create_model(self, model_info):
        """method for creating DDQN Q network"""
        state = Input(shape=self.state_dim, name='states')
        denselayer = Dense(256, activation='relu')(state)
        denselayer = Dense(512, activation='relu')(denselayer)
        denselayer = Dense(512, activation='relu')(denselayer)

        layer_value = Dense(512, activation='relu')(denselayer)
        value = Dense(1, activation=None)(layer_value)

        # advantage
        layer_action = Dense(512, activation='relu')(denselayer)
        action = Dense(self.action_dim, activation=None)(layer_action)

        #mean = Lambda(lambda x: tf.subtract(x, tf.reduce_mean(x, axis=1, keep_dims=True)))(action)
        mean = Lambda(layer_function1)(action)
        #out = Lambda(lambda x: x[0] + x[1])([value, mean])
        out = Lambda(layer_function2)([value, mean])
        model = Model(inputs=state, outputs=out)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

        #model.summary()
        return model


def layer_function1(x):
    """ lambda function """
    return tf.subtract(x, tf.reduce_mean(x, axis=1, keep_dims=True))


def layer_function2(x):
    """ lambda function """
    return x[0] + x[1]
