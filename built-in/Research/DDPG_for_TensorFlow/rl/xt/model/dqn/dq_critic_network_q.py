"""
@Author: Huaqiang Wang, Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

from xt.model.tf_compat import Dense, Input, Model, Adam

from xt.model.dqn.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class DqCriticNetworkQ(XTModel):
    """docstring for ."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        super(DqCriticNetworkQ, self).__init__(model_info)

    def create_model(self, model_info):
        """method for creating DQN Q network"""
        state = Input(shape=self.state_dim)
        denselayer = Dense(HIDDEN1_UNITS, activation='relu')(state)
        denselayer = Dense(HIDDEN2_UNITS, activation='relu')(denselayer)
        value = Dense(self.action_dim, activation='linear')(denselayer)
        model = Model(inputs=state, outputs=value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model
