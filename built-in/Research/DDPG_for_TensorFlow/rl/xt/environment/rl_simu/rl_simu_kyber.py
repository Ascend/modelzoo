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
"""simulation with kyber self-developed"""
import math

from Env_Platform import Env_Platform
from xt.environment.environment import Environment
from xt.framework.register import Registers

EXTENSION_TRACK = 30


@Registers.env.register
class RlEnvSimuKyber(Environment):
    """kybersim environment for lanechange case"""
    def init_env(self, env_info):
        """
        create a kybersim environment instance

        :param: the config information of environment
        :return: the instance of environment
        """
        env_name = env_info["name"]
        vision = env_info["vision"]
        config = env_info["config"]
        env = Env_Platform(env_name, vision, config)

        self.count = 0
        self.reset_count = 0
        self.old_data = None
        self.init_state = None
        return env

    def reset(self, reset_arg=None):
        """
        reset the environment, if there are illegal data in observation
        then use old data

        :param reset_arg: reset scene information
        :return: the observation of environment
        """
        print("!!!!!!!!!!!!  RESET  !!!!!!!!!!!", self.reset_count)
        self.reset_count += 1
        if reset_arg is None:
            state = self.env.reset()
        else:
            state = self.env.reset(reset_arg)
        self.init_state = state
        state[19] = self.reset_end_distance(state[19])
        if self.count == 0:
            self.old_data = state
            self.count += 1
        if self.remove_dirty_data(state) is True:
            state = self.old_data
        self.old_data = state
        return state

    def step(self, action, agent_index=0):
        """
        send lanechange cmd to kybersim

        :param action: action （0-2）
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """
        # print("***step****")
        state, reward, done, info = self.env.step(action)
        state[19] = self.reset_end_distance(state[19])
        if self.remove_dirty_data(state) is True:
            state = self.old_data
        self.old_data = state
        return state, reward, done, info

    @staticmethod
    def reset_end_distance(state):
        """
        to be filled
        """
        state = (
            state[0] - EXTENSION_TRACK,
            state[1] - EXTENSION_TRACK,
            state[2] - EXTENSION_TRACK,
        )

        return state

    @staticmethod
    def remove_dirty_data(state):
        """
        check if state data is illegal

        :param action: state data
        :return: illegal or not
        """
        remove_or_not = False
        for i, _state in enumerate(state):
            if isinstance(state[i], float):
                if math.isnan(state[0]) is True:
                    remove_or_not = True
            if isinstance(state[i], tuple):
                for j in range(len(state[i])):
                    if math.isnan(state[i][j]) is True:
                        remove_or_not = True
            if isinstance(state[i], list):
                for j in range(len(state[i])):
                    if (math.isnan(state[i][j][0]) is True or math.isnan(state[i][j][1]) is True
                            or math.isnan(state[i][j][2]) is True):
                        remove_or_not = True
        return remove_or_not
