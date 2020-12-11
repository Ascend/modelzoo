"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import absolute_import, division, print_function

from xt.model.tf_compat import Dense, Input, Model, Adam, K

from xt.model.impala.default_config import ENTROPY_LOSS, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class ImpalaNetworkMlp(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super(ImpalaNetworkMlp, self).__init__(model_info)

    def create_model(self, model_info):
        """create keras model"""
        state_input = Input(shape=self.state_dim, name='state_input')
        advantage = Input(shape=(1, ), name='adv')
        # old_prediction = Input(shape=(self.action_dim,), name='old_p')

        denselayer = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            denselayer = Dense(HIDDEN_SIZE, activation='tanh')(denselayer)

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
