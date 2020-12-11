"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

import tensorflow as tf
from xt.model.tf_compat import Dense, Input, concatenate, Model, Adam, K
from xt.model.ddpg.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LRC
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class DDPGCriticNetwork(XTModel):
    """docstring for CriticNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LRC
        super(DDPGCriticNetwork, self).__init__(model_info)
        with self.graph.as_default():
            # GRADIENTS for policy update
            K.set_session(self.sess)
            self.action_grads = tf.gradients(self.model.output,
                                             self.action)
            self.sess.run(tf.initialize_all_variables())

    def create_model(self, model_info):
        """create keras model"""
        print("Now we build the model")
        state = Input(shape=self.state_dim[0])
        action = Input(shape=self.state_dim[1], name='action2')
        denselayer0 = Dense(HIDDEN2_UNITS, activation='linear')(action)
        denselayer1 = Dense(HIDDEN1_UNITS, activation='relu')(state)
        denselayer1 = Dense(HIDDEN2_UNITS, activation='linear')(denselayer1)
        denselayer1 = concatenate([denselayer1, denselayer0], axis=-1)
        valuelayer = Dense(HIDDEN2_UNITS, activation='relu')(denselayer1)
        value = Dense(self.action_dim, activation='linear')(valuelayer)
        model = Model(inputs=[state, action], outputs=value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        self.action = action
        self.state = state
        return model

    def gradients(self, states, actions):
        """get gradients for action"""
        with self.graph.as_default():
            return self.sess.run(self.action_grads,
                                 feed_dict={self.state: states,
                                            self.action: actions})[0]
