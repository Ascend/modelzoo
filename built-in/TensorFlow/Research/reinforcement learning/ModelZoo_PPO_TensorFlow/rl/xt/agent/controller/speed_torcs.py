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
"""torcs agent with speed control task"""

from __future__ import division, print_function

import numpy as np
from xt.agent import Agent
from xt.framework.register import Registers


class OU(object):
    """to be filled"""
    @staticmethod
    def function(action, mu_, theta, sigma):
        return theta * (mu_ - action) + sigma * np.random.randn(1)


@Registers.agent
class SpeedTorcs(Agent):
    """Steer Torcs agent."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super(SpeedTorcs, self).__init__(env, alg, agent_config, **kwargs)
        self.o_u = OU()
        self.epsilon = 1.0
        self.episode_count = agent_config.get("episode_count", 100000)

    @staticmethod
    def _calc_reward(new_raw_state):
        speed = new_raw_state["Speed"]
        damage = new_raw_state["Damage"]
        if damage > 0:
            reward = -200
        else:
            reward = speed

        return reward

    @staticmethod
    def _convert_state(raw_obs):
        speed = raw_obs["Speed"]
        angle = raw_obs["Angle"]
        pre_speed = raw_obs["Pre_Speed"]
        pre_dist = raw_obs["Pre_Dist"]

        s_t = np.hstack((speed / 300., angle,
                         pre_speed / 300., min(pre_dist / 200., 1.)))
        return s_t

    def _transform_action(self, predict_action):
        action = predict_action[0][0]
        noise_t = max(self.epsilon, 0) * self.o_u.function(action, 0.0, 0.60, 1.0)
        action = min(action + noise_t, 1.)
        action = max(action, -1.)
        return action

    def calc_custom_evaluate(self):
        trajectory_info = self.get_trajectory()
        total_reward = 0
        pre_action = 0
        d_a_traj = []
        _data_len = len(trajectory_info["done"])
        for index in range(_data_len):
            reward = trajectory_info["reward"][index]
            action = trajectory_info["action"][index]

            total_reward += reward
            d_action = action - pre_action
            d_a_traj.append(d_action)
            pre_action = action

        avg_reward = total_reward / len(trajectory_info["done"])
        smooth = np.sum(np.asarray(d_a_traj)**2)
        smooth = float(np.sqrt(smooth) / len(trajectory_info["done"]))
        return avg_reward - 10 * smooth

    def infer_action(self, state, use_explore):
        """Infer an action with the `state`"""
        s_t = self._convert_state(state)

        if use_explore:  # explore with remote predict
            self.send_explorer.send(s_t)
            action = self.recv_explorer.recv()
        else:  # don't explore, used in evaluate
            action = self.alg.predict(state)

        action = self._transform_action(action)

        # update episode value
        self.epsilon -= 1.0 / self.episode_count

        # update transition data
        self.transition_data.update({
            "cur_state": s_t,
            "action": action,
        })

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """overwrite the functions with special operations"""
        info = []
        damage = next_raw_state["Damage"]
        if damage > 0:
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

    def sync_model(self):
        return ("none")
