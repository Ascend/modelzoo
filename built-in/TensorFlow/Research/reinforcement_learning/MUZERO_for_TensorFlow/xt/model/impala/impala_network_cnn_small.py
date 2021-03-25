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
from __future__ import absolute_import, division, print_function

from xt.model.tf_compat import Dense, Input, Conv2D, \
    Model, Adam, Lambda, Flatten, K, Activation

from xt.model.impala.default_config import ENTROPY_LOSS, LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class ImpalaNetworkCnnSmall(XTModel):
    """model for ImpalaNetworkCnn"""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(ImpalaNetworkCnnSmall, self).__init__(model_info)

    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input')
        state_input_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(state_input)
        # state_input_1 = Lambda(layer_function)(state_input)
        advantage = Input(shape=(1, ), name='adv')
        # old_prediction = Input(shape=(self.action_dim,), name='old_p')

        convlayer = Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same')(state_input_1)
        convlayer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(convlayer)
        print(convlayer)
        convlayer = Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding='valid')(convlayer)

        policy_conv = Conv2D(self.action_dim, (1, 1), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(policy_conv)
        out_actions = Activation(activation='softmax', name='output_actions')(flattenlayer)

        flattenlayer = Flatten()(convlayer)
        out_value = Dense(1, name='output_value')(flattenlayer)
        model = Model(inputs=[state_input, advantage], outputs=[out_actions, out_value])
        losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
        lossweights = {"output_actions": 1.0, "output_value": .25}

        model.compile(optimizer=Adam(lr=LR), loss=losses, loss_weights=lossweights)
        # model.summary()
        return model

    def train(self, state, label):
        with self.graph.as_default():
            # print(type(state[2][0][0]))
            K.set_session(self.sess)
            loss = self.model.fit(x={'state_input': state[0], 'adv': state[1]},
                                  y={
                                      "output_actions": label[0],
                                      "output_value": label[1]
                                  },
                                  batch_size=128,
                                  verbose=0)
            return loss


def layer_function(x):
    return K.cast(x, dtype='float32') / 255.


def impala_loss(advantage):
    """loss for impala"""
    def loss(y_true, y_pred):
        policy = y_pred
        log_policy = K.log(policy + 1e-10)
        entropy = -policy * K.log(policy + 1e-10)
        cross_entropy = -y_true * log_policy
        return K.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy)

    return loss
