#!/usr/bin/env python3
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#

from keras import backend
from keras.utils import control_flow_util
from keras.layers.core import Dropout
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
import npu_ops

def dropout_call(self, inputs, training=None):
    """Make Keras Dropout to execute NPU dropout"""
    if training is None:
        training = backend.learning_phase()

    def dropped_inputs():
        return npu_ops.dropout(
            inputs,
            noise_shape=self._get_noise_shape(inputs),
            seed=self.seed,
            keep_prob=1 - self.rate)

    output = control_flow_util.smart_cond(training,
                                          dropped_inputs,
                                          lambda : array_ops.identity(inputs))

    return output

Dropout.call = dropout_call
