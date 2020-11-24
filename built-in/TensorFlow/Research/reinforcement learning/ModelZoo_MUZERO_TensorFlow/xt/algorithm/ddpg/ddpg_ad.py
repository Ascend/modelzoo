"""
@Desc  : Deep Deterministic Policy Gradient algorithm
"""
from __future__ import division, print_function

import os

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.ddpg.default_config import BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE_FREQ
from xt.algorithm.replay_buffer import ReplayBuffer
from xt.framework.register import Registers
from xt.model import model_builder
from xt.util.common import import_config
from xt.algorithm.ddpg.ddpg import DDPG

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm.register
class DDPGAD(DDPG):
    """Deep Deterministic Policy Gradient algorithm"""

    def predict(self, state):
        """predict for ddpg"""
        inputs = state
        outputs = self.actor.predict(inputs)
        # action = self.actor.predict(state.reshape(1, state.shape[0]))
        return outputs

