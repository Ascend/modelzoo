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
"""CartPole agent for reinforce algorithm"""
from __future__ import division, print_function

from xt.agent import Agent

from xt.framework.register import Registers


@Registers.agent
class CartpoleReinforce(Agent):
    """Cartpole Agent with reinforce algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super(CartpoleReinforce, self).__init__(env, alg, agent_config, **kwargs)

    def infer_action(self, state, use_explore):
        """
        Infer an action with the `state`
        :param state:
        :param use_explore:
        :return: action value
        """
        action = self.alg.predict(state)
        # update transition data
        self.transition_data.update({
            "cur_state": state,
            "action": action,
        })
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        self.transition_data.update({
            "next_state": next_raw_state,
            "reward": reward,
            "done": done,
            "info": info
        })

        return self.transition_data
