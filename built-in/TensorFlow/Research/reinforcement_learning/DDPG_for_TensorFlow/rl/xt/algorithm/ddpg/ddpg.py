# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
@Desc  : Deep Deterministic Policy Gradient algorithm
"""
from __future__ import division, print_function

import os

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.ddpg.default_config import BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE_FREQ
from xt.algorithm.replay_buffer import ReplayBuffer
from xt.framework.register import Registers
from xt.model import model_builder
from xt.util.common import import_config

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm.register
class DDPG(Algorithm):
    """Deep Deterministic Policy Gradient algorithm"""

    def __init__(self, model_info, alg_config, **kwargs):
        import_config(
            globals(), alg_config,
        )
        super(DDPG, self).__init__(
            alg_name="ddpg", model_info=model_info["actor"], alg_config=alg_config
        )

        self.target_actor = model_builder(model_info["actor"])
        self.buff = ReplayBuffer(BUFFER_SIZE)
        self.critic = model_builder(model_info["critic"])
        self.critic_target = model_builder(model_info["critic"])
        self.train_step = 0

    def train(self, **kwargs):
        """train api for DDPG algorithm"""
        gamma = 0.9
        self.train_step += 1

        batch = self.buff.get_batch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[2] for e in batch])

        target_q_values = self.critic_target.predict(
            [new_states, self.target_actor.predict(new_states)]
        )
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + gamma * target_q_values[k]

        loss = self.critic.train([states, actions], y_t)
        a_for_grad = self.actor.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)

        if self.train_step % TARGET_UPDATE_FREQ == 0:
            self._update_target()

        return loss

    def save(self, model_path, model_index):
        """refer to alg"""
        self.actor.save_model(model_path + "actor" + str(model_index).zfill(5))
        self.critic.save_model(model_path + "critic" + str(model_index).zfill(5))

        return [
            "actor" + str(model_index).zfill(5) + ".h5",
            "critic" + str(model_index).zfill(5) + ".h5",
        ]

    def restore(self, model_name, model_weights=None):
        """refer to alg"""
        self.actor.load_model(model_name)
        self.target_actor.load_model(model_name)

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
            data = (
                train_data["cur_state"][index],
                train_data["action"][index],
                train_data["reward"][index],
                train_data["next_state"][index],
                train_data["done"][index],
            )
            buff.add(data)  # Add replay buffer

    def predict(self, state):
        """predict for ddpg"""
        action = self.actor.predict(state.reshape(1, state.shape[0]))
        return action

    def _update_target(self):
        weights = self.actor.get_weights()
        self.target_actor.set_weights(weights)

        weights = self.critic.get_weights()
        self.critic_target.set_weights(weights)
