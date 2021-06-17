# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
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
# ============================================================================
"""Gaussian error linear unit."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import common_flags
from tensorflow.python.framework import ops
from npu_device.npu_device import gen_npu_ops as npu_aicore_ops
from absl import flags

FLAGS=flags.FLAGS


@ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op,grad):
  """ The gradient for fastgelu

  Args:
    op:The fastgelu operations that we are differentiating,which we can us to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the fast_gelu op.

  Returns:
    Gradient with respect to the input of fast_gelu
  """
  return [npu_aicore_ops.fast_gelu_grad(grad,op.inputs[0])]

@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
  """Gaussian Error Linear Unit.

  Original paper: https://arxiv.org/abs/1606.08415
  The approximate version is faster.

  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  if FLAGS.use_fastgelu:
    return npu_aicore_ops.fast_gelu(x)
  else:
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
