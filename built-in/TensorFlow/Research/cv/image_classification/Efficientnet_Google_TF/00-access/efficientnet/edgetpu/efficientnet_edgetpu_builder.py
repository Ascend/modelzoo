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
"""Builder for EfficientNet-EdgeTPU models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import tensorflow.compat.v1 as tf

import efficientnet_builder
import efficientnet_model
import utils

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [127.0, 127.0, 127.0]
STDDEV_RGB = [128.0, 128.0, 128.0]


def efficientnet_edgetpu_params(model_name):
  """Get efficientnet-edgetpu params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-edgetpu-S': (1.0, 1.0, 224, 0.2),
      'efficientnet-edgetpu-M': (1.0, 1.1, 240, 0.2),
      'efficientnet-edgetpu-L': (1.2, 1.4, 300, 0.3),
  }
  return params_dict[model_name]


def efficientnet_edgetpu(width_coefficient=None,
                         depth_coefficient=None,
                         dropout_rate=0.2,
                         survival_prob=0.8):
  """Creates an efficientnet-edgetpu model."""
  blocks_args = [
      'r1_k3_s11_e4_i24_o24_c1_noskip',
      'r2_k3_s22_e8_i24_o32_c1',
      'r4_k3_s22_e8_i32_o48_c1',
      'r5_k5_s22_e8_i48_o96',
      'r4_k5_s11_e8_i96_o144',
      'r2_k5_s22_e8_i144_o192',
  ]
  global_params = efficientnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      survival_prob=survival_prob,
      data_format='channels_last',
      num_classes=1001,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None,
      relu_fn=tf.nn.relu,
      # The default is TPU-specific batch norm.
      # The alternative is tf.layers.BatchNormalization.
      batch_norm=utils.TpuBatchNormalization,  # TPU-specific requirement.
      local_pooling=True,  # for EdgeTPU.
      use_se=False)
  decoder = efficientnet_builder.BlockDecoder()
  return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name.startswith('efficientnet-edgetpu'):
    width_coefficient, depth_coefficient, _, dropout_rate = (
        efficientnet_edgetpu_params(model_name))
    blocks_args, global_params = efficientnet_edgetpu(width_coefficient,
                                                      depth_coefficient,
                                                      dropout_rate)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  logging.info('global_params= %s', global_params)
  logging.info('blocks_args= %s', blocks_args)
  return blocks_args, global_params


def build_model(images,
                model_name,
                training,
                override_params=None,
                model_dir=None,
                fine_tuning=False):
  """A helper functiion to creates a model and returns predicted logits.

  Args:
    images: input images tensor.
    model_name: string, the predefined model name.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.
    model_dir: string, optional model dir for saving configs.
    fine_tuning: boolean, whether the model is used for finetuning.

  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  if not training or fine_tuning:
    if not override_params:
      override_params = {}
    override_params['batch_norm'] = utils.BatchNormalization
  blocks_args, global_params = get_model_params(model_name, override_params)
  if not training or fine_tuning:
    global_params = global_params._replace(batch_norm=utils.BatchNormalization)

  if model_dir:
    param_file = os.path.join(model_dir, 'model_params.txt')
    if not tf.gfile.Exists(param_file):
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      with tf.gfile.GFile(param_file, 'w') as f:
        logging.info('writing to %s', param_file)
        f.write('model_name= %s\n\n' % model_name)
        f.write('global_params= %s\n\n' % str(global_params))
        f.write('blocks_args= %s\n\n' % str(blocks_args))

  with tf.variable_scope(model_name):
    model = efficientnet_model.Model(blocks_args, global_params)
    logits = model(images, training=training)

  logits = tf.identity(logits, 'logits')
  return logits, model.endpoints


def build_model_base(images, model_name, training, override_params=None):
  """A helper functiion to create a base model and return global_pool.

  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined MnasNet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mnasnet_model.GlobalParams.

  Returns:
    features: global pool features.
    endpoints: the endpoints for each layer.

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  blocks_args, global_params = get_model_params(model_name, override_params)

  with tf.variable_scope(model_name):
    model = efficientnet_model.Model(blocks_args, global_params)
    features = model(images, training=training, features_only=True)

  features = tf.identity(features, 'global_pool')
  return features, model.endpoints
