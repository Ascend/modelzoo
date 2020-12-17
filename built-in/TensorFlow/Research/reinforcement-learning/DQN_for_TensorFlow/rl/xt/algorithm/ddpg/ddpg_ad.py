# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

