# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 15:10
# @Author  : w00558981
# @Site    : 
# @File    : train_accuracy_rewrite.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes,ops
from tensorflow.python.ops import array_ops,math_ops,state_ops,variable_scope,weights_broadcast_ops
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions,metric_variable,_aggregate_across_replicas


def accuracy(labels,
             predictions,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  """Calculates how often `predictions` matches `labels`.

  The `accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    predictions: The predicted values, a `Tensor` of any shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.accuracy is not supported when eager '
                       'execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  if labels.dtype != predictions.dtype:
    predictions = math_ops.cast(predictions, labels.dtype)
  is_correct = math_ops.cast(
      math_ops.equal(predictions, labels), dtypes.float32)
  return mean(is_correct, weights, metrics_collections, updates_collections,
              name or 'accuracy')



def mean(values,
         weights=None,
         metrics_collections=None,
         updates_collections=None,
         name=None):
  """Computes the (weighted) mean of the given values.

  The `mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A `Tensor` representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean is not supported when eager execution '
                       'is enabled.')

  with variable_scope.variable_scope(name, 'mean', (values, weights)):
    values = math_ops.cast(values, dtypes.float32)

    total = metric_variable([], dtypes.float32, name='total')
    count = metric_variable([], dtypes.float32, name='count')

    if weights is None:
      num_values = math_ops.cast(array_ops.size(values), dtypes.float32)
    else:
      values, _, weights = _remove_squeezable_dimensions(
          predictions=values, labels=None, weights=weights)
      weights = weights_broadcast_ops.broadcast_weights(
          math_ops.cast(weights, dtypes.float32), values)
      values = math_ops.multiply(values, weights)
      num_values = math_ops.reduce_sum(weights)

    update_total_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)

    def compute_mean(_, t, c):
      return math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')

    mean_t = _aggregate_across_replicas(
        metrics_collections, compute_mean, total, count)

    #### modified to current value
    update_op = math_ops.reduce_sum(values)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_t, update_op



