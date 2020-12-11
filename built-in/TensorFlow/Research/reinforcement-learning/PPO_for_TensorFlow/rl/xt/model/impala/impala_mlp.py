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
from xt.model.tf_compat import Dense, Input, Model, Adam, K
from xt.model.impala.default_config import ENTROPY_LOSS, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model
class ImpalaMlp(XTModel):
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super().__init__(model_info)

    def create_model(self, model_info):
        """create keras model"""
        state_input = Input(shape=self.state_dim, name='state_input')
        advantage = Input(shape=(1, ), name='adv')

        denselayer = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        for _ in range(NUM_LAYERS - 1):
            denselayer = Dense(HIDDEN_SIZE, activation='relu')(denselayer)

        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions')(denselayer)  # y_pred
        out_value = Dense(1, name='output_value')(denselayer)
        model = Model(inputs=[state_input, advantage], outputs=[out_actions, out_value])
        losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
        lossweights = {"output_actions": 1.0, "output_value": .5}

        model.compile(optimizer=Adam(lr=LR), loss=losses, loss_weights=lossweights)
        return model

    def train(self, state, label):
        with self.graph.as_default():
            # print(type(state[2][0][0]))
            K.set_session(self.sess)
            loss = self.model.fit(x={'state_input': state[0], 'adv': state[1]},
                                  y={
                                      "output_actions": label[0],  # y_ture
                                      "output_value": label[1]
                                  },
                                  batch_size=128,
                                  verbose=0)
            return loss


def impala_loss(advantage):
    """loss for impala"""
    def loss(y_true, y_pred):
        policy = y_pred
        log_policy = K.log(policy + 1e-10)
        entropy = (-policy * log_policy)
        cross_entropy = (-y_true * log_policy)
        return K.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy)

    return loss
