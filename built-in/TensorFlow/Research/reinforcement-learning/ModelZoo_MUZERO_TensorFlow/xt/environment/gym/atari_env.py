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
"""atari env for simulation"""
import numpy as np

from xt.environment.environment import Environment
from xt.environment.gym.atari_wrappers import make_atari
from xt.framework.register import Registers


@Registers.env.register
class AtariEnv(Environment):
    """It encapsulates an openai gym environment."""
    def init_env(self, env_info):
        """
        create a atari environment instance

        :param: the config information of environment.
        :return: the instance of environment
        """
        name = env_info.get("name")
        dim=env_info.get('dim', 84)
        stack_size = env_info.get('stack_size', 4)
        grayscale = env_info.get('grayscale', True)
        
        if not grayscale:
            stack_size *= 3
        self.stackedobs = np.zeros((dim, dim, stack_size), np.uint8)

        self.init_state = None
        self.done = True

        return make_atari(name, dim, grayscale)

    def reset(self):
        """
        reset the environment, if done is true, must clear obs array

        :return: the observation of gym environment
        """
        if self.done:
            obs = self.env.reset()
            self.stackedobs[...] = 0
            self.stackedobs[..., -obs.shape[-1]:] = obs
        state = self.stackedobs
        self.init_state = state
        return state

    def step(self, action, agent_index=0):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        if done:
            self.stackedobs[...] = 0

        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs
        state = self.stackedobs
        self.done = done
        return state, reward, done, info
