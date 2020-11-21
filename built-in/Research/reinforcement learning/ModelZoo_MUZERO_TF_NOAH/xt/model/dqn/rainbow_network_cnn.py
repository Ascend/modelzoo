"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from xt.model.tf_compat import Dense, Input, Conv2D, \
    Model, Adam, Lambda, Flatten, K

from xt.model.dqn.default_config import LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class RainbowNetworkCnn(XTModel):
    """docstring for ."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.atoms = 51
        super(RainbowNetworkCnn, self).__init__(model_info)

    def create_model(self, model_info):
        """create keras model"""
        state = Input(shape=self.state_dim, name='state_input')
        action = Input(shape=(2, ), name='action', dtype='int32')
        target_p = Input(shape=(self.atoms, ), name="target_p")
        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='same')(state)
        convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(512, activation='relu')(flattenlayer)
        value = Dense(1, activation=None)(denselayer)

        denselayer = Dense(512, activation='relu')(flattenlayer)
        atom = Dense(self.action_dim * self.atoms, activation=None)(denselayer)
        mean = Lambda(lambda x: tf.subtract(
            tf.reshape(x, [-1, self.action_dim, self.atoms]),
            tf.reduce_mean(tf.reshape(x, [-1, self.action_dim, self.atoms]), axis=1, keep_dims=True)))(atom)

        value = Lambda(lambda x: tf.add(tf.expand_dims(x[0], 1), x[1]))([value, mean])
        #prob = Lambda(lambda x: tf.nn.softmax(x), name="output")(value)
        #pylint error Lambda may not be necessary
        prob = tf.nn.softmax(value, name="output")
        model = Model(inputs=[state, action, target_p], outputs=prob)

        adam = Adam(lr=self.learning_rate, clipnorm=10.)
        model.compile(loss=[dist_dqn_loss(action=action, target_p=target_p)], optimizer=adam)

        return model

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)
            # print(type(state[2][0][0]))
            loss = self.model.fit(x={
                'state_input': state[0],
                'action': state[1],
                'target_p': state[2]
            },
                                  y={"output": label},
                                  verbose=0)
            return loss


def dist_dqn_loss(action, target_p):
    """loss for rainbow"""
    def loss(y_true, y_pred):
        y_pred = tf.gather_nd(y_pred, action)
        return -K.mean(target_p * K.log((y_pred + 1e-10)))

    return loss
