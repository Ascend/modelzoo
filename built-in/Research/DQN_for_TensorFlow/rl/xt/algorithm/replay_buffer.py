"""
@Author: Jack Qian, chenchen
@license : Copyright(C), Huawei
"""
from __future__ import division, print_function

import random
from collections import deque


class ReplayBuffer(object):
    """ReplayBuffer class"""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        """Randomly sample batch_size examples"""
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)

        return random.sample(self.buffer, int(batch_size))

    def size(self):
        """get buffer size"""
        return self.buffer_size

    def add(self, train_data):
        """put data to buffer"""
        experience = train_data
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        """
        if buffer is full, return buffer size
        otherwise, return experience counter
        """
        return self.num_experiences

    def erase(self):
        """remove data from buffer"""
        self.buffer = deque()
        self.num_experiences = 0
