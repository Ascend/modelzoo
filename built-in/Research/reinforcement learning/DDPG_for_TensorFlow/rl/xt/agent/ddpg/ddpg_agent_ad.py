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
"""CartPole agent for dqn algorithm"""
from __future__ import division, print_function

import random
import numpy as np

from xt.agent import Agent

from xt.framework.register import Registers
from xt.framework.comm.message import message



@Registers.agent.register
class DdpgAgentAD(Agent):
    """DDPG Agent with dqn algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.o_u = OU()
        self.epsilon = 1.0
        self.episode_count = agent_config.get("episode_count", 100000)
        self.batch_size = alg.alg_config["BATCH_SIZE"]

    def infer_action(self, state, use_explore):
        """Infer an action with the `state`"""
        s_t = state
        state_dim = state.shape[0]

        if use_explore:  # explore with remote predict
            expand_state = np.ones((self.batch_size, state_dim))
            expand_state[0] = state
            send_data = message(expand_state, cmd="predict")
            # print("ddpg agent send data:", send_data)
            self.send_explorer.send(send_data)
            action = self.recv_explorer.recv()[0]
        else:  # don't explore, used in evaluate
            expand_state = np.ones((self.batch_size, state_dim))
            expand_state[0] = state
            action = self.alg.predict(expand_state)[0]

        action = self._transform_action(action)

        # update episode value
        self.epsilon -= 1.0 / self.episode_count

        # update transition data
        self.transition_data.update({
            "cur_state": s_t,
            "action": action,
        })
        # print('action', action)
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """overwrite the functions with special operations"""
        info = []
        # print("reward", reward, done)
        self.transition_data.update({
            "next_state": next_raw_state,
            "reward": reward,
            "done": done,
            "info": info
        })

        # deliver this transition data to learner, trigger train process.
        if use_explore:
            train_data = {k: [v] for k, v in self.transition_data.items()}
            # train_data = {self.id: train_data}
            train_data = message(train_data, agent_id=self.id)
            self.send_explorer.send(train_data)

        return next_raw_state

    def sync_model(self):
        return ("none")

    def _transform_action(self, predict_action):
        action = predict_action[0]
        noise_t = max(self.epsilon, 0) * self.o_u.function(action, 0.0, 0.60, 1.0)
        action = min(action + noise_t, [1.])
        action = max(action, [-1.])
        return action

class OU(object):
    """to be filled"""
    @staticmethod
    def function(action, mu_, theta, sigma):
        return theta * (mu_ - action) + sigma * np.random.randn(1)
