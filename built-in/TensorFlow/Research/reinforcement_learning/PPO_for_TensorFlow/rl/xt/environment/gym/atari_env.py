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
import cv2
import numpy as np
from collections import deque

from xt.environment.environment import Environment
from xt.environment.gym.atari_wrappers import make_atari
from xt.framework.register import Registers

cv2.ocl.setUseOpenCL(False)

@Registers.env
class AtariEnv(Environment):
    """It encapsulates an openai gym environment."""
    def init_env(self, env_info):
        """
        create a atari environment instance

        :param: the config information of environment.
        :return: the instance of environment
        """
        self.dim = env_info.get('dim', 84)
        self.grayscale = env_info.get('grayscale', True)

        if self.grayscale:
            self.init_obs = np.zeros((self.dim, self.dim, 1), np.uint8)
        else:
            self.init_obs = np.zeros((self.dim, self.dim, 3), np.uint8)
        self.stack_size = 4
        self.stack_obs = deque(maxlen=self.stack_size)
        self.init_stack_obs(self.stack_size)

        self.init_state = None
        self.done = True

        return make_atari(env_info)

    def init_stack_obs(self, num):
        for _ in range(num):
            self.stack_obs.append(self.init_obs)

    def reset(self):
        """
        reset the environment, if done is true, must clear obs array

        :return: the observation of gym environment
        """
        if self.done:
            obs = self.env.reset()
            self.init_stack_obs(self.stack_size - 1)
            self.stack_obs.append(self.obs_proc(obs))

        state = np.concatenate(self.stack_obs, -1)
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
            self.init_stack_obs(self.stack_size - 1)

        self.stack_obs.append(self.obs_proc(obs))
        # self.stack_obs.append(obs)
        state = np.concatenate(self.stack_obs, -1)
        self.done = done
        return state, reward, done, info

    def obs_proc(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.dim, self.dim), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        return obs

@Registers.env
class VectorAtariEnv(Environment):
    """vector atari environment to speedup"""
    def init_env(self, env_info):
        """create multi-env as a vector """
        self.vector_env_size = env_info.get("vector_env_size")
        assert self.vector_env_size is not None, "vector env must assign 'env_num'."

        self.env_vector = list()
        for _ in range(self.vector_env_size):
            self.env_vector.append(AtariEnv(env_info))

    def reset(self):
        """reset each env within vector"""
        state = [env.reset() for env in self.env_vector]
        self.init_state = state

        return state

    def step(self, action, agent_index=0):
        """
        step in order.
        :param action:
        :param agent_index:
        :return:
        """
        batch_obs, batch_reward, batch_done, batch_info = list(), list(), list(), list()
        for env_id in range(self.vector_env_size):
            obs, reward, done, info = self.env_vector[env_id].step(action[env_id])
            if done:
                obs = self.env_vector[env_id].reset()

            batch_obs.append(obs)
            batch_reward.append(reward)
            batch_done.append(done)
            batch_info.append(info)

        return batch_obs, batch_reward, batch_done, batch_info

    def get_env_info(self):
        """return environment's basic information,
        vector environment only support single agent now.
        """
        self.reset()
        env_info = {
            "n_agents": self.n_agents,
            "api_type": self.api_type,
        }
        agent_ids = [0]
        env_info.update({"agent_ids": agent_ids})

        return env_info

    def close(self):
        [env.close() for env in self.env_vector]
