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
"""
Agent module for multi-agents case.
Could  inherit from the single agent, with fetch special state/action etc.
"""

from xt.agent.ppo.cartpole_ppo import CartpolePpo
from xt.framework.register import Registers


@Registers.agent
class CatchPigsPpo(CartpolePpo):
    """catch pigs agent for ppo share weights"""
    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """
        handle env feedback with current agent.id
        :param next_raw_state:
        :param reward:
        :param done:
        :param info:
        :param use_explore:
        :return:
        """

        super().handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        # add next state for unified api within multi-agents
        self.transition_data.update({
            "next_state": next_raw_state})

        return self.transition_data
