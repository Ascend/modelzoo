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
"""muzero algorithm """
import random

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.muzero.default_config import BATCH_SIZE, BUFFER_SIZE, GAMMA, TD_STEP, UNROLL_STEP
from xt.algorithm.replay_buffer import ReplayBuffer
from xt.framework.register import Registers
from xt.util.common import import_config


@Registers.algorithm.register
class Muzero(Algorithm):
    """ muzero algorithm """
    def __init__(self, model_info, alg_config, **kwargs):
        """
        Algorithm instance, will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        import_config(globals(), alg_config)
        super().__init__(
            alg_name=kwargs.get("name") or "muzero",
            model_info=model_info["actor"],
            alg_config=alg_config,
        )
        self.buff = ReplayBuffer(BUFFER_SIZE)
        self.discount = GAMMA
        self.unroll_step = UNROLL_STEP
        self.td_step = TD_STEP
        self.async_flag = False

    def train(self, **kwargs):
        """ muzero train process."""
        trajs = self.buff.get_batch(BATCH_SIZE)
        traj_pos = [(t, self.sample_position(t)) for t in trajs]
        traj_data = [(g["cur_state"][i], g["action"][i:i + self.unroll_step],
                      self.make_target(i, g)) for (g, i) in traj_pos]
        image = np.asarray([e[0] for e in traj_data])
        actions = np.asarray([e[1] for e in traj_data])
        targets = [e[2] for e in traj_data]

        target_values = []
        target_rewards = []
        target_policys = []
        for target in targets:
            target_values.append([e[0] for e in target])
            target_rewards.append([e[1] for e in target])
            target_policys.append([e[2] for e in target])

        target_values = np.asarray(target_values)
        target_rewards = np.asarray(target_rewards)
        target_policys = np.asarray(target_policys)
        loss = self.actor.train([image, actions],
                                [target_values, target_rewards, target_policys])

        return loss

    def prepare_data(self, train_data, **kwargs):
        if len(train_data["reward"]) > self.unroll_step + 1:
            self.buff.add(train_data)

    def sample_position(self, traj):
        return random.randint(0, len(traj["reward"]) - self.unroll_step-1)

    def make_target(self, state_index, traj):
        """Generate targets to learn from during the network training."""

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        root_values = traj["root_value"]
        rewards = traj["reward"]
        child_visits = traj["child_visits"]
        for current_index in range(state_index, state_index + self.unroll_step + 1):
            bootstrap_index = current_index + self.td_step
            if bootstrap_index < len(root_values):
                value = root_values[bootstrap_index] * self.discount ** self.td_step
            else:
                value = 0

            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(root_values):
                targets.append((value, rewards[current_index], child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets
