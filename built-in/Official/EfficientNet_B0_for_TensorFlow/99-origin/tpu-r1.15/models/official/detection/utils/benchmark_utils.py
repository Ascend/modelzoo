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
"""Benchmark utils for detection models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def compute_model_statistics(batch_size, is_training=True):
  """Compute number of parameters and FLOPS."""
  num_trainable_params = np.sum(
      [np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
  tf.logging.info('number of trainable params: {}'.format(num_trainable_params))

  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  flops_per_image = flops / batch_size
  if is_training:
    tf.logging.info(
        'number of FLOPS per image: {} in training'.format(flops_per_image))
  else:
    tf.logging.info(
        'number of FLOPS per image: {} in eval'.format(flops_per_image))
