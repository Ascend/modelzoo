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
"""sumo dqn with static shape"""

from __future__ import division, print_function

import os
import numpy as np

from xt.framework.register import Registers
from xt.algorithm.dqn.dqn import DQN

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm
class DQNSUMO(DQN):
    """Deep Q learning algorithm.
    """
    def predict(self, state):
        """The api will call the keras.model.predict as default,
        if the inputs is different from the normal state,
        You need overwrite this function."""

        inputs = state
        #print("[----------predict_shape-----------]:", inputs.shape)
        out = self.actor.predict(inputs)
        #print("[-------expand_values--------]:", out,  out.shape)
        return np.argmax(out, 1)
