#!/usr/bin/env python
# coding=utf-8

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

import six

from tensorflow.python.ops import math_ops
from tensorflow.keras.metrics import Mean
from keras import backend as K
from tensorflow.python.framework import dtypes
from keras.utils import losses_utils
from keras.utils import metrics_utils
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.keras.metrics import get
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


class MeanMetricWrapper(Mean):
  """Wraps a stateless metric function with the Mean metric
  Args:
    fn: The metric function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    **kwargs: The keyword arguments that are passed on to 'fn'.
  """

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.
    `y_true` and `y_pred` should have the same shape.
    Args:
      y_true: Ground truth values, shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      sample_weight: Optional `sample_weight` acts as a
        coefficient for the metric. If a scalar is provided, then the metric is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the metric for each sample of the batch is rescaled
        by the corresponding element in the `sample_weight` vector. If the shape
        of `sample_weight` is `[batch_size, d0, .., dN-1]` (or can be broadcasted 
        to this shape), then each metric element of `y_pred` is scaled by the 
        corresponding value of `sample_weight`. (Note on `dN-1`: all metric 
        functions reduce by 1 dimension, usually the last axis (-1)).
    Returns:
      Update op.   
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    [y_true, y_pred], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
    matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {}

    if type(self) is MeanMetricWrapper:
      config['fn'] = self._fn

    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # Note that while MeanMetricWrapper itself isn't public, object of this 
    # class may be created and added to the model by calling model.compile
    fn = config.pop('fn', None)
    if cls is MeanMetricWrapper:
      return cls(get(fn), **config)
    return super(MeanMetricWrapper, cls).from_config(config)

def categorical_accuracy_int32(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
        math_ops.argmax(y_true, axis=-1, output_type=dtypes.int32),
        math_ops.argmax(y_pred, axis=-1, output_type=dtypes.int32)),
      K.floatx())

def sparse_categorical_accuracy_int32(y_true, y_pred):
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = ops.convert_to_tensor_v2_with_dispatch(y_true)
  y_pred_rank = y_pred.shape.ndims
  y_true_rank = y_true.shape.ndims
  
  if(y_true_rank is not None) and (y_pred_rank is not None) and (len(K.int_shape(y_true))==len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true,[-1])
  y_pred = math_ops.argmax(y_pred,axis = -1, output_type = dtypes.int32)
  if K.dtype(y_pred) != K.dtype(y_true):
    y_pred = math_ops.cast(y_pred,K.dtype(y_true))
  return math_ops.cast(math_ops.equal(y_true,y_pred),K.floatx())


class SparseCategoricalAccuracyInt32(MeanMetricWrapper):
  def __init__(self, name='sparse_categorical_accuracy_int32', dtype=None):
    super(SparseCategoricalAccuracyInt32, self).__init__(sparse_categorical_accuracy_int32, name, dtype=dtype)
  