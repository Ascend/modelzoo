#!/usr/bin/env python
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
"""torcs agent with steer control task"""
from __future__ import division, print_function

import numpy as np

from xt.agent.controller.speed_torcs import SpeedTorcs
from xt.framework.register import Registers


@Registers.agent
class SteerTorcs(SpeedTorcs):
    """Steer_Torcs Agent."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super(SteerTorcs, self).__init__(env, alg, agent_config, **kwargs)

    @staticmethod
    def _calc_reward(new_raw_state):
        sp_ = new_raw_state["Speed"]
        damage = new_raw_state["Damage"]
        trackpos = new_raw_state["ToMiddle"]
        angle = new_raw_state["Angle"]
        if damage > 0:
            reward = -200
        else:
            reward = sp_ * np.cos(angle) - np.abs(sp_ * np.sin(angle))
            reward = reward * (1 - abs(trackpos))

        return reward

    @staticmethod
    def _convert_state(raw_obs):
        speed = raw_obs["Speed"]
        angle = raw_obs["Angle"]
        trackpos = raw_obs["ToMiddle"]

        s_t = np.hstack((speed / 300., angle, trackpos))
        return s_t

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        info = []
        damage = next_raw_state["Damage"]
        if damage > 0:
            done = True

        trackpos = next_raw_state["ToMiddle"]
        if trackpos > 1. or trackpos < -1.:
            done = True

        self.transition_data.update({
            "next_state": self._convert_state(next_raw_state),
            "reward": self._calc_reward(next_raw_state),
            "done": done,
            "info": info
        })

        # deliver this transition data to learner, trigger train process.
        if use_explore:
            train_data = {k: [v] for k, v in self.transition_data.items()}
            train_data = {self.id: train_data}
            self.send_explorer.send(train_data)

        return next_raw_state
