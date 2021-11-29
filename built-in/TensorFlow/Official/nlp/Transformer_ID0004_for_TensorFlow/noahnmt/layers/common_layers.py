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

import six
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import function


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(
          ([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t


def reverse_sequence(input, seq_lengths, seq_dim, batch_dim):
  if seq_lengths is not None:
    return tf.reverse_sequence(
        input=input, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)
  else:
    return tf.reverse(input, axis=[seq_dim])


def split_last_dim(x, n):
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])

def combine_last_two_dims(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])

def unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.nn.rnn_cell.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]


def shard_features(features, data_parallelism):
  """ split features along axis 0, i.e., batch dim

  return: a list of features with sub-tensor
  """
  num_shards = data_parallelism.n
  if num_shards <= 1:
    return [features]
  
  # {key:value} to {key:[value1,value2,...]}
  sharded_features = dict()
  for k, v in sorted(six.iteritems(features)):
    v = tf.convert_to_tensor(v)
    v_shape = shape_list(v)
    if not v_shape:
      v = tf.expand_dims(v, axis=-1)
      v_shape = [1]
    if v_shape == [1]:
      v = tf.tile(v, tf.to_int32([num_shards]))
    sharded_features[k] = data_parallelism(
        tf.identity, tf.split(v, num_shards, 0))
  
  # {key:[value1 value2 ...]} to [{key:value1}, {key:value2} ...]}
  datashard_features = []
  assert len(sharded_features[list(features.keys())[0]]) == num_shards
  for d in range(num_shards):
    f = {k: v[d] for k, v in six.iteritems(sharded_features)}
    datashard_features.append(f)
  return datashard_features


def summarize_features(features, num_shards=1):
  """Generate summaries for features."""

  with tf.name_scope("input_stats"):
    tf.summary.scalar("num_sents", tf.shape(features["source_ids"])[0])
    tf.summary.scalar("num_source_words", tf.reduce_sum(features["source_mask"]))
    if "target_len" in features:
      tf.summary.scalar("num_target_words", tf.reduce_sum(features["target_mask"]))


def average_sharded_losses(sharded_losses):
  """Average losses across datashards.

  Args:
    sharded_losses: list<dict<str loss_name, Tensor loss>>. The loss
      can be a single Tensor or a 2-tuple (numerator and denominator).

  Returns:
    losses: dict<str loss_name, Tensor avg_loss>
  """
  losses = {}
  for loss_name in sorted(sharded_losses[0]):
    all_shards = [shard_losses[loss_name] for shard_losses in sharded_losses]
    if isinstance(all_shards[0], tuple):
      sharded_num, sharded_den = zip(*all_shards)
      mean_loss = (
          tf.add_n(sharded_num) / tf.maximum(
              tf.cast(1, sharded_num[0].dtype), 
              tf.cast(tf.add_n(sharded_den), sharded_num[0].dtype)))
    else:
      mean_loss = tf.reduce_mean(all_shards)

    losses[loss_name] = mean_loss
  return losses


def remove_summaries():
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                       x.device, cast_x.device)
  return cast_x


def print_trainable_variables(mode=""):
  tf.logging.info("================================")
  tf.logging.info("==== Params in %s mode ====" % mode)
  for var in tf.trainable_variables():
    tf.logging.info("%s: %s: %s" % (var.name, var.get_shape(), var.device))
  tf.logging.info("Total num of params: " + str(np.sum([
      np.prod(var.get_shape().as_list())
      for var in tf.trainable_variables()
  ])))
  tf.logging.info("================================")





################## for NPU ###################



@tf.custom_gradient
def gather_npu(params, indices):
  def grad(dy):
    params_shape = tf.shape(params, out_type=tf.int64)
    params_shape = tf.cast(params_shape, tf.int32)
    #dy = tf.cast(dy, tf.float32)
    grad_gather = tf.unsorted_segment_sum(dy, indices, params_shape[0])
    return grad_gather, None
  return tf.gather(params, indices), grad


# from tensorflow.contrib.offline_train.python.npu_unary_ops import npu_unary_ops
# from tensorflow.contrib.offline_train.python import npu_ops
from npu_bridge.estimator import npu_ops

def npu_dropout(input_tensor, dropout_prob):
  # return input_tensor
  return npu_ops.dropout(input_tensor, 1.0 - dropout_prob)


