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
"""rainbow algorithm, compound of dqn, c51 etc."""
from __future__ import absolute_import, division, print_function

import os

import numpy as np

from xt.algorithm.dqn.default_config import BATCH_SIZE, BUFFER_SIZE, GAMMA, TARGET_UPDATE_FREQ
from xt.algorithm.dqn.dqn import DQN
from xt.algorithm.prioritized_replay_buffer import PrioritizedReplayBuffer
from xt.framework.register import Registers
from xt.model.tf_compat import loss_to_val
from xt.util.common import import_config

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm.register
class Rainbow(DQN):
    """rainbow algorithm"""
    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super(Rainbow, self).__init__(model_info, alg_config, **kwargs)
        self.buff = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=0.6)
        self.multi_step = alg_config.get('multi_step', 1)
        self.data_list = dict()
        self.gamma = GAMMA**self.multi_step
        self.v_min = -10
        self.v_max = 10
        self.atoms = 51
        self.z_lin = np.linspace(self.v_min, self.v_max, self.atoms)
        self.delta_z = self.z_lin[1] - self.z_lin[0]

    def train(self, **kwargs):
        """rainbow train process"""
        buff = self.buff
        self.train_count += 1

        batch = buff.sample(BATCH_SIZE, 0.4)
        states, actions, rewards, new_states, dones, weights, batch_idxes = batch

        target_dist = self.get_target_dist(new_states, rewards, dones)
        fake_label = np.zeros((len(rewards), self.action_dim, self.atoms))
        loss = self.actor.train([states, actions, target_dist], fake_label)

        new_priorities = np.abs(loss_to_val(loss)) + 1e-6
        new_priorities = np.repeat(new_priorities, len(batch_idxes))
        # print(len(new_priorities))
        buff.update_priorities(batch_idxes, new_priorities)
        if self.train_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return loss

    def get_target_dist(self, new_states, rewards, dones):
        """target dist calculation"""
        batch = rewards.shape[0]
        batch_idx = np.arange(batch)

        actions = np.zeros((batch, 1)).astype(int)
        target_p = np.zeros((batch, self.atoms))
        q_tp1 = self.actor.predict([new_states, actions, target_p])
        q_tp1 = np.sum(q_tp1 * self.z_lin, -1)
        q_tp1_best = np.argmax(q_tp1, 1)
        target_dist_q = self.target_actor.predict([new_states, actions, target_p])

        target_dist_q_ = np.zeros((batch, self.atoms))
        for i in range(batch):
            target_dist_q_[i] = target_dist_q[i][q_tp1_best[i]]

        target_dist = np.zeros((batch, self.atoms))
        atoms_att = dict()
        for i in range(self.atoms):
            atoms_att['z'] = rewards + (1 - dones) * self.z_lin[i] * self.gamma
            atoms_att['z'] = np.clip(atoms_att['z'], self.v_min, self.v_max)
            atoms_att['pos'] = (atoms_att['z'] - self.v_min) / self.delta_z
            atoms_att['ub'] = np.ceil(atoms_att['pos']).astype(int)
            atoms_att['lb'] = np.floor(atoms_att['pos']).astype(int)

            target_dist[batch_idx, atoms_att['lb']] += \
                (atoms_att['ub'] - atoms_att['pos']) * target_dist_q_[batch_idx, i]
            target_dist[batch_idx, atoms_att['ub']] += \
                (atoms_att['pos'] - atoms_att['lb']) * target_dist_q_[batch_idx, i]

        return target_dist

    def prepare_data(self, train_data, **kwargs):
        """
        prepare the train data for dqn,
        here, just add once new data into replay buffer.
        :param train_data:
        :return:
        """
        data_len = len(train_data["done"])
        for index in range(data_len):
            if self.multi_step == 1:
                self.buff.add(train_data["cur_state"][index],
                              train_data["action"][index],
                              train_data["reward"][index],
                              train_data["next_state"][index],
                              float(train_data["done"][index]))  # Add replay buffer

    def cum_reward(self, reward_list):
        """accumulation reward with gamma"""
        reward = 0.
        for rew in reward_list[::-1]:
            reward += rew * self.gamma
        return reward

    def predict(self, state):
        """overwrite DQN with the special predict operation"""
        inputs = state.reshape((1, ) + state.shape)
        action = np.zeros((1, 1)).astype(int)
        target_q = np.zeros((1, self.atoms))
        out = self.actor.predict([inputs, action, target_q])
        out = np.sum(out[0] * self.z_lin, 1)
        # print(Q.shape)
        return np.argmax(out)
