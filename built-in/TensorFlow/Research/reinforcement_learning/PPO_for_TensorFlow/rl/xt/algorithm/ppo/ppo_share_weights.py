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
ppo share weights among multi-agents.
"""

from xt.algorithm.ppo.ppo import PPO
from xt.framework.register import Registers


@Registers.algorithm
class PPOShareWeights(PPO):
    def __init__(self, model_info, alg_config, **kwargs):
        """
        Algorithm instance, will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        super(PPOShareWeights, self).__init__(model_info=model_info,
                                              alg_config=alg_config,
                                              name="ppo_share_weights")
        # update weights map for multi-agent catch the pig
        # self.weights_map = {"0": {"prefix": "actor"}, "1": {"prefix": "actor"}}
        # update weights map for multi-agent figure 8
        self.weights_map = {"rl_0": {"prefix": "actor"}, "rl_1": {"prefix": "actor"}, "rl_2": {"prefix": "actor"}, \
            "rl_3": {"prefix": "actor"}, "rl_4": {"prefix": "actor"}, "rl_5": {"prefix": "actor"}, "rl_6": {"prefix": "actor"}, \
            "rl_7": {"prefix": "actor"}, "rl_8": {"prefix": "actor"}, "rl_9": {"prefix": "actor"}}
