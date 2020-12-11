"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

import tensorflow as tf
from xt.model.tf_compat import K, Conv2D, Dense, Flatten, Input, Lambda, Model, Adam

from xt.model import XTModel
from xt.model.ppo.default_config import ENTROPY_LOSS, LOSS_CLIPPING, LR
from xt.util.common import import_config
from xt.framework.register import Registers


@Registers.model
class ActorCriticPPOCnn(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(ActorCriticPPOCnn, self).__init__(model_info)

    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input')
        #state_input_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(state_input)
        state_input_1 = Lambda(layer_function)(state_input)
        advantage = Input(shape=(1, ), name='adv')
        old_prediction = Input(shape=(self.action_dim, ), name='old_p')
        old_value = Input(shape=(1, ), name='old_v')

        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='same', name='con_1')(state_input_1)
        convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same', name='con_2')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='con_3')(convlayer)
        flatten_layer = Flatten(name='flatten_1')(convlayer)
        denselayer = Dense(512, activation='relu', name='dense_1')(flatten_layer)
        # make layer's name dummy for get partial weights
        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions_raw')(denselayer)
        out_value = Dense(1, name='output_value')(denselayer)
        model = Model(inputs=[state_input, advantage, old_prediction, old_value], outputs=[out_actions, out_value])
        losses = {
            "output_actions_raw": ppo_loss(advantage=advantage, old_prediction=old_prediction),
            "output_value": value_loss(old_value),
        }
        lossweights = {"output_actions_raw": 1.0, "output_value": .5}

        model.compile(optimizer=Adam(lr=LR), loss=losses, loss_weights=lossweights)
        return model

    def train(self, state, label):
        with self.graph.as_default():
            # print(type(state[2][0][0]))
            K.set_session(self.sess)
            loss = self.model.fit(x={
                'state_input': state[0],
                'adv': state[1],
                'old_p': state[2],
                'old_v': state[3]
            },
                                  y={
                                      "output_actions_raw": label[0],
                                      "output_value": label[1]
                                  },
                                  verbose=0)
            return loss


def layer_function(x):
    return K.cast(x, dtype='float32') / 255.


def value_loss(old_value):
    """value loss for ppo"""
    def loss(y_true, y_pred):
        vpredclipped = old_value + K.clip(y_pred - old_value, -LOSS_CLIPPING, LOSS_CLIPPING)
        # Unclipped value
        vf_losses1 = K.square(y_pred - y_true)
        # Clipped value
        vf_losses2 = K.square(vpredclipped - y_true)

        vf_loss = .5 * K.mean(K.maximum(vf_losses1, vf_losses2))

        return vf_loss

    return loss


def ppo_loss(advantage, old_prediction):
    """loss for ppo"""
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        ratio = prob / (old_prob + 1e-10)
        return -K.mean(
            K.minimum(ratio * advantage,
                      K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) +
            ENTROPY_LOSS * (prob * K.log((prob + 1e-10))))

    return loss


def ppo_loss_new(advantage, old_prediction):
    """loss for ppo"""
    def loss(y_true, y_pred):
        neglogpac = -y_true * K.log(y_pred)
        old_neglog = -y_true * K.log(old_prediction)
        ratio = tf.exp(old_neglog - neglogpac)

        return -K.mean(
            K.minimum(ratio * advantage,
                      K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) -
            ENTROPY_LOSS * (y_pred * K.log((y_pred + 1e-10))))

    return loss
