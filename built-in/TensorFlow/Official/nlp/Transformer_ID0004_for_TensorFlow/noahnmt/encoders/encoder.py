# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
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
Abstract base class for encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import six

from noahnmt.configurable import Configurable
from noahnmt.graph_module import GraphModule


ENCODER_OUTPUT = "encoder_output"
FINAL_STATE = "final_state"
ENCODER_MEM = "encoder_mem"


@six.add_metaclass(abc.ABCMeta)
class Encoder(GraphModule, Configurable):
  """Abstract encoder class. All encoders should inherit from this.

  Args:
    params: A dictionary of hyperparameters for the encoder.
    name: A variable scope for the encoder graph.
  """

  def __init__(self, params, mode, name):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)

  def _build(self, inputs, *args, **kwargs):
    return self.encode(inputs, *args, **kwargs)

  @abc.abstractmethod
  def encode(self, inputs, sequence_length, **kwargs):
    """
    Encodes an input sequence.

    Args:
      inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
      sequence_length: The length of each input. An int32 tensor of shape [T].

    Returns:
      An `EncoderOutput` tuple containing the outputs and final state.
    """
    raise NotImplementedError
