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
"""Define metrics for the UNet 3D Model."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import tensorflow as tf


def dice(y_true, y_pred, axis=(1, 2, 3, 4)):
  """DICE coefficient.

  Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
      analysis, selection, and tool. BMC Med Imaging. 2015;15:29. Published
      2015
      Aug 12. doi:10.1186/s12880-015-0068-x

  Implemented according to
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ6

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z, num_classes].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z, num_classes].
    axis: axises of features.

  Returns:
    DICE coefficient.
  """
  y_true = tf.cast(y_true, y_pred.dtype)
  eps = tf.keras.backend.epsilon()

  intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
  summation = tf.reduce_sum(
      y_true, axis=axis) + tf.reduce_sum(
          y_pred, axis=axis)
  return (2 * intersection + eps) / (summation + eps)


def generalized_dice(y_true, y_pred, axis=(1, 2, 3)):
  """Generalized Dice coefficient, for multi-class predictions.

  For output of a multi-class model, where the shape of the output is
  (batch, x, y, z, n_classes), the axis argument should be (1, 2, 3).

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z, num_classes].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z, num_classes].
    axis: axises of features.

  Returns:
    DICE coefficient.
  """
  y_true = tf.cast(y_true, y_pred.dtype)

  if y_true.get_shape().ndims < 2 or y_pred.get_shape().ndims < 2:
    raise ValueError('y_true and y_pred must be at least rank 2.')

  epsilon = tf.keras.backend.epsilon()
  w = tf.math.reciprocal(tf.square(tf.reduce_sum(y_true, axis=axis)) + epsilon)
  num = 2 * tf.reduce_sum(
      w * tf.reduce_sum(y_true * y_pred, axis=axis), axis=-1)
  den = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=axis), axis=-1)
  return (num + epsilon) / (den + epsilon)


def hamming(y_true, y_pred, axis=(1, 2, 3)):
  """Hamming distance.

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z].
    axis: a list, axises of the feature dimensions.
  Return: Hamming distance value.
  """
  y_true = tf.cast(y_true, y_pred.dtype)
  return tf.reduce_mean(tf.not_equal(y_pred, y_true), axis=axis)


def jaccard(y_true, y_pred, axis=(1, 2, 3, 4)):
  """Jaccard Similarity.

  Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
      analysis, selection, and tool. BMC Med Imaging. 2015;15:29. Published
      2015
      Aug 12. doi:10.1186/s12880-015-0068-x

  Implemented according to
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ7

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z, num_classes].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z, num_classes].
    axis: axises of features.

  Returns:
    Jaccard similarity.
  """
  y_true = tf.cast(y_true, y_pred.dtype)
  eps = tf.keras.backend.epsilon()

  intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
  union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
  return (intersection + eps) / (union - intersection + eps)


def tversky(y_true, y_pred, axis=(1, 2, 3), alpha=0.3, beta=0.7):
  """Tversky similarity.

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z, num_classes].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z, num_classes].
    axis: axises of spatial dimensions.
    alpha: weight of the prediction.
    beta: weight of the groundtruth.

  Returns:
    Tversky similarity coefficient.
  """
  y_true = tf.cast(y_true, y_pred.dtype)

  if y_true.get_shape().ndims < 2 or y_pred.get_shape().ndims < 2:
    raise ValueError('y_true and y_pred must be at least rank 2.')

  eps = tf.keras.backend.epsilon()

  num = tf.reduce_sum(y_pred * y_true, axis=axis)
  den = (
      num + alpha * tf.reduce_sum(y_pred * (1 - y_true), axis=axis) +
      beta * tf.reduce_sum((1 - y_pred) * y_true, axis=axis))
  # Sum over classes.
  return tf.reduce_sum((num + eps) / (den + eps), axis=-1)


def adaptive_dice32(y_true, y_pred, data_format='channels_last'):
  """Adaptive dice metric.

  Args:
    y_true: the ground truth matrix. Shape [batch_size, x, y, z, num_classes].
    y_pred: the prediction matrix. Shape [batch_size, x, y, z, num_classes].
    data_format: channel last of channel first.

  Returns:
    Adaptive dice value.
  """
  epsilon = 10**-7
  y_true = tf.cast(y_true, dtype=y_pred.dtype)
  # Determine axes to pass to tf.reduce_sum
  if data_format == 'channels_last':
    ndim = len(y_pred.shape)
    reduction_axes = list(range(ndim - 1))
  else:
    reduction_axes = 1

  # Calculate intersections and unions per class
  intersections = tf.reduce_sum(y_true * y_pred, axis=reduction_axes)
  unions = tf.reduce_sum(y_true + y_pred, axis=reduction_axes)

  # Calculate Dice scores per class
  dice_scores = 2.0 * (intersections + epsilon) / (unions + epsilon)

  # Calculate weights based on Dice scores
  weights = tf.exp(-1.0 * dice_scores)

  # Multiply weights by corresponding scores and get sum
  weighted_dice = tf.reduce_sum(weights * dice_scores)

  # Calculate normalization factor
  norm_factor = tf.size(dice_scores, out_type=tf.float32) * tf.exp(-1.0)

  weighted_dice = tf.cast(weighted_dice, dtype=tf.float32)

  # Return 1 - adaptive Dice score
  return 1 - (weighted_dice / norm_factor)
