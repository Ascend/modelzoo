# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import time
import numpy as np
import tensorflow as tf
from xt.model.tf_compat import K, Conv2D, Dense, \
    Flatten, Input, Lambda, Model, Activation

from xt.model import XTModel
from xt.model.ppo.default_config import ENTROPY_LOSS, LOSS_CLIPPING, LR
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model
class PPOCnnTf(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super().__init__(model_info)
        self.total_time = 0
        self.total_train_time = 0
        self.train_count = 0

    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input')
        #state_input_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(state_input)
        state_input_1 = Lambda(layer_function)(state_input)
        adv = Input(shape=(1, ), name='adv')
        old_p = Input(shape=(self.action_dim, ), name='old_p')
        old_v = Input(shape=(1, ), name='old_v')

        convlayer = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', padding='same')(state_input_1)
        convlayer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(convlayer)
        # print(convlayer)
        convlayer = Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding='valid')(convlayer)

        policy_conv = Conv2D(self.action_dim, (1, 1), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(policy_conv)
        out_actions = Activation(activation='softmax', name='output_actions_raw')(flattenlayer)

        flattenlayer = Flatten()(convlayer)
        out_value = Dense(1, name='output_value')(flattenlayer)

        model = Model(inputs=[state_input], outputs=[out_actions, out_value])

        self.state = state_input
        self.adv = adv
        self.old_p = old_p
        self.old_v = old_v
        self.out_p = out_actions
        self.out_v = out_value
        self.target_v = tf.placeholder(tf.float32, name="target_value",
                                       shape=(None, 1))

        self.target_p = tf.placeholder(tf.float32, name="target_policy",
                                       shape=(None, self.action_dim))

        loss = 0.5 * value_loss(self.target_v, self.out_v, self.old_v)
        loss += ppo_loss(self.adv, self.old_p, self.target_p, self.out_p)
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(LR)
        self.train_op = self.optimizer.minimize(loss)
        self.sess.run(tf.initialize_all_variables())

        return model

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)
            nbatch = state[0].shape[0]
            nbatch_train = 128
            inds = np.arange(nbatch)
            loss_val = []
            start_time = time.time()
            for _ in range(4):
                # Randomize the indexes

                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    obs = state[0][mbinds]
                    adv = state[1][mbinds]
                    old_p = state[2][mbinds]
                    old_v = state[3][mbinds]
                    target_p = label[0][mbinds]
                    target_v = label[1][mbinds]
                    start_train_time = time.time()
                    feed_dict = {self.state: obs,
                                 self.adv: adv,
                                 self.old_p: old_p,
                                 self.old_v: old_v,
                                 self.target_p: target_p,
                                 self.target_v: target_v}
                    ret_value = self.sess.run([self.train_op, self.loss], feed_dict)
                    self.total_train_time += time.time() - start_train_time
                    # loss_val.append(np.mean(ret_value[1]))
            self.total_time += time.time() - start_time
            self.train_count += 1
            if self.train_count % 100 == 0:
                print("total_time", self.total_time/self.train_count, "total_train_time", self.total_train_time/self.train_count)
            # print(loss_val)
            return np.mean(loss_val)

    def predict(self, state):
        """
        Do predict use the newest model.
        :param state:
        :return:
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            feed_dict = {self.state: state}
            return self.sess.run([self.out_p, self.out_v], feed_dict)

def layer_function(x):
    return K.cast(x, dtype='float32') / 255.

def value_loss(target_v, out_v, old_v):
    vpredclipped = old_v + K.clip(out_v - old_v, -LOSS_CLIPPING, LOSS_CLIPPING)
    # Unclipped value
    vf_losses1 = K.square(out_v - target_v)
    # Clipped value
    vf_losses2 = K.square(vpredclipped - target_v)

    vf_loss = .5 * K.mean(K.maximum(vf_losses1, vf_losses2))

    return vf_loss

def ppo_loss(adv, old_p, target_p, out_p):
    """loss for ppo"""
    neglogpac = -target_p * K.log(out_p + 1e-10)
    old_neglog = -target_p * K.log(old_p + 1e-10)
    ratio = tf.exp(old_neglog - neglogpac)

    return -K.mean(
        K.minimum(ratio * adv,
                  K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * adv) -
        ENTROPY_LOSS * (out_p * K.log((out_p + 1e-10))), 1)
