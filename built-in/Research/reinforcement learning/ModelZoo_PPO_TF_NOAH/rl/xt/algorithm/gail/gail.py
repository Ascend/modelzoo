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
"""GAIL algorithm."""
import numpy as np
from pyarrow import deserialize

import lz4.frame
from xt.algorithm import Algorithm, alg_builder
from xt.framework.register import Registers
from xt.util.common import import_config
from xt.util.data import get_data, get_datalen, init_file


@Registers.algorithm
class GAIL(Algorithm):
    """IMPALA algorithm"""
    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super(GAIL, self).__init__(alg_name="gail", model_info=model_info["gail"],
                                   alg_config=alg_config)

        alg_para = {}
        alg_para.update({'alg_name': alg_config['alg_name']})
        alg_para.update({'async_flag': alg_config['async_flag']})
        alg_para.update({'model_info': model_info})
        alg_para.update({'alg_config': alg_config})
        # print(alg_para)
        self.policy_alg = alg_builder(**alg_para)
        if alg_config.get("type", 'learner') != 'actor':
            expert_data_path = alg_config.get("expert_data", None)
            self.expert_data = init_file(expert_data_path)
            self.expert_data_len = get_datalen(self.expert_data)
            self.data_index = 0
        self.labels = np.zeros((1280, 2))
        self.state = None
        self.action = None

    def train(self, **kwargs):
        """refer to alg"""
        actor_loss = self.policy_alg.train()

        expert_state, expert_action = self.get_expert_data()
        data_len = min(self.state.shape[0], expert_state.shape[0])
        if data_len <= 1:
            return 0.
        # print('====data len====', data_len, self.state.shape[0], expert_state.shape[0])
        self.actor.train(
            [self.state[0:data_len], self.action[0:data_len],
             expert_state[0:data_len], expert_action[0:data_len]],
            self.labels[0:data_len])
        return actor_loss

    def save(self, model_path, model_index):
        """use policy alg to save"""
        actor_name = self.policy_alg.save(model_path, model_index)

        return actor_name

    def restore(self, model_name, model_weights=None):
        self.policy_alg.load(model_name, model_weights)

    def get_weights(self):
        """use sub policy alg to get weights"""
        return self.policy_alg.get_weights()

    def prepare_data(self, train_data, **kwargs):
        # episode_data = train_data[1]
        # get reward from gail
        states = np.asarray(train_data["cur_state"])
        actions = np.asarray(train_data["real_action"])
        action_matrix = np.eye(self.action_dim)[actions.reshape(-1)]
        # new_reward = self.actor.predict([states, action_matrix, states, action_matrix])
        # fixme: gail algorithm not ready  # pylint: disable=W0511
        # for i, data in enumerate(train_data):
        #     data[2] = new_reward[i]

        # push new data to polcy algorithm
        self.policy_alg.prepare_data(train_data)
        self.state = states
        self.action = action_matrix

    def output(self, state):
        """refer to alg"""
        pred = self.policy_alg.output(state)

        return pred

    def get_expert_data(self):
        data = get_data(self.expert_data, self.data_index)
        data = deserialize(lz4.frame.decompress(data))
        episode_data = data[1]
        states = np.asarray([e[0] for e in episode_data]) * 256
        states = states.astype('int8')
        actions = np.asarray([e[1] for e in episode_data])
        actions = np.eye(self.action_dim)[actions.reshape(-1)]
        self.data_index = (self.data_index + 1) % self.expert_data_len
        return states, actions
