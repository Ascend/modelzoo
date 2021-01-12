"""
@Author: Huaqing Wang, Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

import tensorflow as tf
from xt.model.tf_compat import Dense, Input, Model, K
from xt.model.ddpg.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LRA
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class DDPGActorNetwork(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LRA
        super(DDPGActorNetwork, self).__init__(model_info)
        with self.graph.as_default():
            K.set_session(self.sess)
            self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
            self.params_grad = tf.gradients(self.model.output, self.weights,
                                            -self.action_gradient)
            grads = zip(self.params_grad, self.weights)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
            self.sess.run(tf.initialize_all_variables())

    def create_model(self, model_info):
        state = Input(shape=self.state_dim)
        denselayer = Dense(HIDDEN1_UNITS, activation='relu')(state)
        denselayer = Dense(HIDDEN2_UNITS, activation='relu')(denselayer)
        action = Dense(1, activation='tanh')(denselayer)

        model = Model(inputs=state, outputs=action)
        self.weights = model.trainable_weights
        self.state = state
        return model

    def train(self, states, action_grads):
        with self.graph.as_default():
            self.sess.run(self.optimize,
                          feed_dict={self.state: states,
                                     self.action_gradient: action_grads})
            return self.sess.run(self.params_grad,
                                 feed_dict={self.state: states,
                                            self.action_gradient: action_grads})
