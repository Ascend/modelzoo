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
"""sumo log utils"""

import random

import numpy as np


class ReplayBuffer:
    """ReplayBuffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ResultRecorder:
    """record the result of each episode"""
    def __init__(self, history_length=1000):
        self.history_length = history_length
        self.buffer = [[], [], []]
        self.lengths = [0, 0, 0]
        self.succ_time = 0
        self.succ_num = 0

    def add_result(self, result, direction):  #0 left ; 1 straight ; 2 right
        self.buffer[direction].append(result - 1)  # 1: fail ; 2: success
        self.lengths[direction] += 1
        if len(self.buffer[direction]) > self.history_length:
            self.buffer[direction].pop(0)

    def mean(self, direction):
        return np.mean(self.buffer[direction])

    def length(self, direction):
        return self.lengths[direction]

    def succ_mean_time(self):
        return self.succ_time / max(float(self.succ_num), 1)


class RewardRecorder:
    """record the reward info of the dqn experiments"""
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)

    @property
    def mean(self):
        return np.mean(self.buffer)

    # get the length of total episodes
    @property
    def num_episodes(self):
        return self._episode_length
