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
"""dqn algorithm with prioritized replay"""
from __future__ import division, print_function

import os

import numpy as np

from xt.algorithm.dqn.default_config import BATCH_SIZE, BUFFER_SIZE, GAMMA, TARGET_UPDATE_FREQ
from xt.algorithm.dqn.dqn import DQN
from xt.algorithm.prioritized_replay_buffer import PrioritizedReplayBuffer
from xt.framework.register import Registers

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm
class DQNPri(DQN):
    """DQN with prioritized replay buffer."""
    def __init__(self, model_info, alg_config, **kwargs):
        super(DQNPri, self).__init__(model_info, alg_config, **kwargs)
        self.buff = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=0.6)

    def train(self, **kwargs):
        self.train_count += 1

        batch = self.buff.sample(BATCH_SIZE, 0.4)
        train_dict = dict()
        (train_dict['states'], train_dict['actions'], train_dict['rewards'],
         train_dict['new_states'], train_dict['dones'], train_dict['weights'],
         train_dict['batch_idxes']) = batch

        train_dict['y_t'] = self.actor.predict(train_dict['states'])
        train_dict['target_q_values'] = self.target_actor.predict(train_dict['new_states'])
        predict = self.target_actor.predict(train_dict['states'])
        maxq = np.max(train_dict['target_q_values'], 1)
        td_errors = []

        for k in range(len(train_dict['states'])):
            if train_dict['dones'][k]:
                q_value = train_dict['rewards'][k]
            else:
                q_value = train_dict['rewards'][k] + GAMMA * maxq[k]
            train_dict['y_t'][k][train_dict['actions'][k]] = q_value
            td_error = train_dict['y_t'][k][train_dict['actions'][k]] - \
                       predict[k][train_dict['actions'][k]]
            td_errors.append(td_error)

        loss = self.actor.train(train_dict['states'], train_dict['y_t'])
        new_priorities = np.abs(td_errors) + 1e-6
        self.buff.update_priorities(train_dict['batch_idxes'], new_priorities)

        if self.train_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return loss

    def prepare_data(self, train_data, **kwargs):
        """
        prepare the train data for dqn,
        here, just add once new data into replay buffer.
        :param train_data:
        :return:
        """
        buff = self.buff
        data_len = len(train_data["done"])
        for index in range(data_len):
            buff.add(train_data["cur_state"][index], train_data["action"][index],
                     train_data["reward"][index], train_data["next_state"][index],
                     train_data["done"][index])  # Add replay buffer
