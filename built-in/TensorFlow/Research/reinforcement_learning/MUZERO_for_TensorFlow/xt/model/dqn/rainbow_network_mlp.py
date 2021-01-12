"""
@Author: Huaqiang Wang,
@license : Copyright(C), Huawei
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from xt.model.tf_compat import Dense, Input, Lambda, Model, Adam, K
from xt.model.dqn.default_config import HIDDEN1_UNITS, LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class RainbowNetworkMlp(XTModel):
    """docstring for ."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.atoms = 51
        super(RainbowNetworkMlp, self).__init__(model_info)

    def create_model(self, model_info):
        """create keras model"""
        state = Input(shape=self.state_dim, name='state_input')
        action = Input(shape=(1, ), name='action', dtype='int32')
        target_p = Input(shape=(self.atoms, ), name="target_p")
        denselayer = Dense(HIDDEN1_UNITS, activation='relu')(state)

        layer_value = Dense(self.atoms, activation=None)(denselayer)
        layer_a = Dense(self.action_dim * self.atoms, activation=None)(denselayer)

        mean = Lambda(lambda x: tf.subtract(
            tf.reshape(x, [-1, self.action_dim, self.atoms]),
            tf.reduce_mean(tf.reshape(x, [-1, self.action_dim, self.atoms]), axis=1, keep_dims=True)))(layer_a)

        value = Lambda(lambda x: tf.add(tf.expand_dims(x[0], 1), x[1]))([layer_value, mean])
        output = Lambda(lambda x: tf.nn.softmax(tf.reshape(x, [-1, self.action_dim, self.atoms])),
                        name="output")(value)

        model = Model(inputs=[state, action, target_p], outputs=output)

        adam = Adam(lr=self.learning_rate, clipnorm=10.)
        model.compile(loss=[dist_dqn_loss(action=action, target_p=target_p)], optimizer=adam)

        return model

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)
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
        batch_dim = tf.shape(action)[0]
        action_idx = tf.reshape(action, [batch_dim])
        cat_idx = tf.transpose(tf.reshape(tf.concat([tf.range(batch_dim), action_idx], axis=0), [2, batch_dim]))
        errors = tf.reduce_sum(-target_p * K.log(tf.gather_nd(y_pred, cat_idx) + 1e-10), axis=-1)
        return tf.reduce_mean(errors)

    return loss
