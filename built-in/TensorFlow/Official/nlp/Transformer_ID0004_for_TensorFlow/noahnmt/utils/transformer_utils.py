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
utilities used in transformer network
https://arxiv.org/abs/1706.03762.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import pdb
import numpy as np

from noahnmt.utils import constant_utils
from noahnmt.layers import common_layers
from noahnmt.utils import graph_utils


def activation_swish(x, beta=1.0):
  return x * tf.nn.sigmoid(beta * x)

def _relative_attention_inner(x, z, transpose):
  """Relative position-aware dot-product attention inner calculation.
  This batches matrix multiply calculations to avoid unnecessary broadcasting.
  Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [length, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.
  Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
  """
  batch_size = tf.shape(x)[0]
  heads = x.get_shape().as_list()[1]
  length = tf.shape(x)[2]

  # x_t is [length, batch_size, heads, length or depth]
  x_t = tf.transpose(x, [2, 0, 1, 3])
  # x_t_r is [length, batch_size * heads, length or depth]
  x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
  # x_tz_matmul is [length, batch_size * heads, length or depth]
  x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
  # x_tz_matmul_r is [length, batch_size, heads, length or depth]
  x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
  # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
  x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
  return x_tz_matmul_r_t

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



# fp16 
# def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
#   """Layer normalize the tensor x, averaging over the last dimension."""
#   # epsilon = 1e-4 if x.dtype == tf.float16 else 1e-6
    
#   x_dtype = x.dtype
#   x = tf.cast(x, dtype=tf.float32)
  
#   # the last dim
#   if filters is None:
#     filters = x.get_shape()[-1]
#   with tf.variable_scope(
#       name, default_name="layer_norm", 
#       values=[x], reuse=reuse):
#     # gamma and beta used in layer_norm
#     scale = tf.get_variable(
#         "g", [filters], 
#         initializer=tf.ones_initializer(
#             dtype=x.dtype),
#         dtype=x.dtype)
#     bias = tf.get_variable(
#         "b", [filters], 
#         initializer=tf.zeros_initializer(
#             dtype=x.dtype),
#         dtype=x.dtype)
    
#     # layer norm based on mean and variance
#     mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
#     variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
#     norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

#     y = norm_x * scale + bias
#     y = tf.cast(y, x_dtype)
#     return y
# fp16 

from tensorflow.contrib.framework.python.ops import add_arg_scope
@add_arg_scope
def contrib_layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
  """Adds a Layer Normalization layer.
  Based on the paper:
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    https://arxiv.org/abs/1607.06450.
  Can be used as a normalizer function for conv2d and fully_connected.
  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
  if requested, is performed over axes `begin_params_axis .. R - 1`.
  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
  meaning that normalization is performed over all but the first axis
  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
  parameters are calculated for the rightmost axis (the `C` if `inputs` is
  `NHWC`).  Scaling and recentering is performed via broadcast of the
  `beta` and `gamma` parameters with the normalized tensor.
  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
  and this part of the inputs' shape must be fully defined.
  Args:
    inputs: A tensor having rank `R`. The normalization is performed over
      axes `begin_norm_axis ... R - 1` and centering and scaling parameters
      are calculated over `begin_params_axis ... R - 1`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    begin_norm_axis: The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    scope: Optional scope for `variable_scope`.
  Returns:
    A `Tensor` representing the output of the operation, having the same
    shape and dtype as `inputs`.
  Raises:
    ValueError: If the rank of `inputs` is not known at graph build time,
      or if `inputs.shape[begin_params_axis:]` is not fully defined at
      graph build time.
  """
  from tensorflow.contrib.framework.python.ops import variables
  from tensorflow.contrib.layers.python.layers import utils
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import init_ops
  from tensorflow.python.ops import nn
  from tensorflow.python.ops import variable_scope
  with variable_scope.variable_scope(
      scope, 'layer_norm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    #graph_utils.add_dict_to_collection({"inputs1": inputs}, "SAVE_TENSOR")
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
          collections=beta_collections,
          trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma = variables.model_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          collections=gamma_collections,
          trainable=trainable)
    
    gamma = tf.cast(gamma, tf.float32)
    beta = tf.cast(beta, tf.float32)
    inputs = tf.cast(inputs, tf.float32)

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_input": inputs}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_gamma": gamma}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_beta": beta}, "SAVE_TENSOR")

    # Calculate the moments on the last axis (layer activations).
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_mean": mean}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_variance": variance}, "SAVE_TENSOR")

    #graph_utils.add_dict_to_collection({"mean": mean}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({"variance": variance}, "SAVE_TENSOR")
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1e-6 #if dtype != tf.float16 else 1e-3

    #gamma = tf.reshape(gamma, [1, 1, inputs_shape[-1]])
    #beta = tf.reshape(beta, [1, 1, inputs_shape[-1]])
    
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon,
        name="batchnorm_nofusion")
    
    outputs = tf.cast(outputs, dtype)

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_output": outputs}, "SAVE_TENSOR")

    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)



def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  # x_shape = common_layers.shape_list(x)
  # if len(x_shape) > 2:
  #   x = tf.reshape(x, [-1, x_shape[-1]])
  #graph_utils.add_dict_to_collection({"encoder_emb_processed3": x}, "SAVE_TENSOR")
  #x = contrib_layer_norm(x, begin_norm_axis=-1, begin_params_axis=-1)
  #x=tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=-1, begin_params_axis=-1)
  
  # if len(x_shape) > 2:
  #   x = tf.reshape(x, x_shape)
  # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_output": scale}, "SAVE_TENSOR")
  #return x
  
  
  
  # epsilon = 1e-4 if x.dtype == tf.float16 else 1e-6
  #return contrib_layer_norm(x, begin_norm_axis=-1, begin_params_axis=-1)
  # return tf.keras.layers.LayerNormalization(axis=-1,epsilon=1e-3)(x)

  x_dtype = x.dtype  
  x = tf.cast(x, dtype=tf.float32)

  shape_list = common_layers.shape_list(x)
  batch = shape_list[0]
  length = shape_list[1]

  # the last dim
  if filters is None:
    filters = shape_list[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", 
      values=[x], reuse=reuse):
    # gamma and beta used in layer_norm
    scale = tf.get_variable(
        "gamma", [filters], 
        initializer=tf.ones_initializer(
            dtype=tf.float32),
        dtype=tf.float32)
    # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/scaleeeee": scale}, "SAVE_TENSOR")
    bias = tf.get_variable(
        "beta", [filters], 
        initializer=tf.zeros_initializer(
            dtype=tf.float32),
        dtype=tf.float32)
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_input": x}, "SAVE_TENSOR")
    

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_scale": scale}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_bias": bias}, "SAVE_TENSOR")
    
    # reshape to avoid error
    scale = tf.reshape(scale, [1,filters])
    bias = tf.reshape(bias, [1,filters])
    # scale = tf.tile(scale, [batch * length, 1])
    # bias = tf.tile(bias, [batch * length, 1])

    # layer norm based on mean and variance
    #mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    #variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
    with tf.name_scope("batch_norm_custom"):
      epsilon_tensor = tf.reshape(tf.constant([epsilon]), [1,1])
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon_tensor)
    y = norm_x * scale + bias
    
    y = tf.cast(y, x_dtype)

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_output": y}, "SAVE_TENSOR")

    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_after_layer_norm": y}, "SAVE_TENSOR")

    return y




def dropout_residual_layer(x, y, dropout_rate, is_training, name=None):
  with tf.variable_scope(
      name, default_name="dropout_residual"): 
    # dropout
    if is_training:
      # y = tf.layers.dropout(
      #     inputs=y, 
      #     rate=dropout_rate,
      #     training=is_training)
      y = common_layers.npu_dropout(
          input_tensor=y,
          dropout_prob=dropout_rate)
    # residual
    x = x + y
    return x

def position_encoding(length, depth,
                      min_timescale=1,
                      max_timescale=1e4):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    depth: an int

  Returns:
    Tensor of shape (length, depth)
  """
  # depth = depth // 2
#   depth = float(depth)
  positions = np.arange(length, dtype=np.float32)
  # correspond to log(10000^(1/(d-1)))
  log_timescale_increment = (
      np.log(max_timescale / min_timescale) / (depth - 1))
  # correspond to 1 / 10000^(i/(d-1)), i=0....d-1
  inv_timescales = min_timescale * np.exp(
      np.arange(depth, dtype=np.float32) * -log_timescale_increment)
  # pos / 10000^(i/(d-1))
  scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
  # intead of using SIN and COS interleaved
  # it's  the same to first use SIN then COS
  # as they are applied to the same position
  x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
  return tf.constant(x, dtype=tf.float32)



def single_position_encoding(pos, depth,
                            min_timescale=1,
                            max_timescale=1e4):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    depth: an int

  Returns:
    Tensor of shape (2*depth)
  """
  position = tf.cast(pos, constant_utils.DT_FLOAT())
  # correspond to log(10000^(1/(d-1)))
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (depth - 1))
  # correspond to 1 / 10000^(i/(d-1)), i=0....d-1
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(depth), constant_utils.DT_FLOAT()) * -log_timescale_increment)
  # pos / 10000^(i/(d-1))
  scaled_time = position * inv_timescales
  # intead of using SIN and COS interleaved
  # it's  the same to first use SIN then COS
  # as they are applied to the same position
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=0)


# def compute_qkv(query,
#                 memory,
#                 key_depth,
#                 value_depth):
#   """Computes query, key and value.

#   Args:
#     query_antecedent: a Tensor with shape [batch, length_q, channels]
#     memory_antecedent: a Tensor with shape [batch, length_m, channels]
#     total_key_depth: an integer
#     total_value_depth: and integer
#     q_filter_width: An integer specifying how wide you want the query to be.
#     kv_filter_width: An integer specifying how wide you want the keys and values
#     to be.
#     q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
#     kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

#   Returns:
#     q, k, v : [batch, length, depth] tensors
#   """
#   if memory is None:
#     memory = query
  
#   def _compute(inp, depth, filter_width, padding, name):
#       return tf.layers.dense(inp, depth, use_bias=False, name=name)

#   q = tf.layers.dense(query, key_depth, use_bias=False, name="q")
#   k = tf.layers.dense(memory, key_depth, use_bias=False, name="k")
#   v = tf.layers.dense(memory, value_depth, use_bias=False, name="v")
#   return q, k, v


def precompute_encdec_kv(memory,
                        key_depth,
                        value_depth,
                        num_heads,
                        batch_size,
                        seq_len):
  """precompute key and value.
  """
  # kv = tf.layers.dense(memory, key_depth + value_depth, use_bias=False, name="kv")
  # k, v = tf.split(kv, [key_depth, value_depth], axis=-1)
  graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_memory_prekv": memory}, "SAVE_TENSOR")
  k = tf.layers.dense(memory, key_depth, use_bias=False, name="k")
  v = tf.layers.dense(memory, value_depth, use_bias=False, name="v")
  graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_k_prekv": k}, "SAVE_TENSOR")
  graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_v_prekv": v}, "SAVE_TENSOR")
  # reshape to [batch, heads, time, units/heads]
  # k = split_heads(k, num_heads)
  # v = split_heads(v, num_heads)
  k = tf.reshape(k, [batch_size, seq_len, num_heads, key_depth//num_heads])
  k = tf.transpose(k, [0, 2, 1, 3])
  v = tf.reshape(v, [batch_size, seq_len, num_heads, value_depth//num_heads])
  v = tf.transpose(v, [0, 2, 1, 3])
  return k, v



def compute_qkv_fused(query,
                      memory,
                      key_depth,
                      value_depth,
                      ignore_kv=False):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: and integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory is None:
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_query": query}, "SAVE_TENSOR")
    #pdb.set_trace()
    q = tf.layers.dense(query, key_depth, use_bias=False, name="q")
    k = tf.layers.dense(query, key_depth, use_bias=False, name="k")
    v = tf.layers.dense(query, value_depth, use_bias=False, name="v")

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_query_dense": q}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_key_dense": k}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_value_dense": v}, "SAVE_TENSOR")
    # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_after_dense": qkv}, "SAVE_TENSOR")

    # q, k, v = tf.split(qkv, [key_depth, key_depth, value_depth], axis=-1)
  else:

    q = tf.layers.dense(query, key_depth, use_bias=False, name="q")

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_query_dense": q}, "SAVE_TENSOR")
    
    if ignore_kv:
      return q

    k = tf.layers.dense(memory, key_depth, use_bias=False, name="k")
    v = tf.layers.dense(memory, value_depth, use_bias=False, name="v")

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_k_dense": k}, "SAVE_TENSOR")
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_v_dense": v}, "SAVE_TENSOR")

    # k, v = tf.split(kv, [key_depth, value_depth], axis=-1)
  return q, k, v


def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """

  def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.

    The first of these two dimensions is n.

    Args:
      x: a Tensor with shape [..., m]
      n: an integer.

    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
      assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])
  x = split_last_dimension(x, num_heads)
  return tf.transpose(x, [0, 2, 1, 3])


def dot_product_attention(q,
                          k,
                          v,
                          mask,
                          dropout_rate,
                          is_training,
                          edge_k=None,
                          edge_v=None,
                          name=None):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string

  Returns:
    A Tensor.
  """
  #pdb.set_trace()
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # query shape: [batch, num_heads, query_length, units]
    # key shap: [batch, num_heads, mem_length, units]
    # scores shape: [batch, num_heads, query_length, memory_length]
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_qqqqqq": q}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_kkkkkk": k}, "SAVE_TENSOR")
    scores = tf.matmul(q, k, transpose_b=True)
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_scores": scores}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({scores.name: scores}, "SAVE_TENSOR")

    if edge_k is not None:
      scores += _relative_attention_inner(q, edge_k, True)

    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_scores": scores}, "SAVE_TENSOR")

    dtype = scores.dtype
    # scores = tf.cast(scores, tf.float32)
    if mask is not None:
        # mask shape [batch, query_length, mem_length]
        # mshape = tf.shape(mask)
        # assert len(mask.get_shape().as_list()) == 3
        mask = tf.expand_dims(mask, 1)
        if len(mask.get_shape().as_list()) == 3:
            mask = tf.expand_dims(mask, 1)
                
        # cast back to fp32 if in mixed-precision training mode     
        # if dtype != tf.float32:
        #     scores = tf.cast(x=scores, dtype=tf.float32)
        #     mask = tf.cast(x=mask, dtype=tf.float32)
        #     scores = scores * mask + ((1.0 - mask) * -constant_utils.INF) # score dtype: fp32
        # else:
        #     scores = scores * mask + ((1.0 - mask) * -constant_utils.INF) # score dtype: fp32
        #pdb.set_trace()
        if mask.dtype != scores.dtype:
          mask = tf.cast(mask, scores.dtype)
        adder = (1.0-mask) * -constant_utils.INF

        # adder = tf.cast(adder, tf.float32)

        graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_adder": adder}, "SAVE_TENSOR")

        scores = scores + adder
    
    # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_masked_scores_fp16": scores}, "SAVE_TENSOR")
        
    scores = tf.cast(scores, tf.float32)

    # scores = scores + adder

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_masked_scores": scores}, "SAVE_TENSOR")

    # scores_shape = common_layers.shape_list(scores)
    # scores = tf.reshape(scores, [-1, scores_shape[-1]])
    
    scores_normalized = tf.nn.softmax(scores, name="score_norm") # compute softmax in fp32

    # scores_normalized = tf.reshape(scores_normalized, scores_shape)
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_softmax": scores_normalized}, "SAVE_TENSOR")

    if dtype != tf.float32:
       scores_normalized = tf.cast(scores_normalized, dtype) # convert back to fp16 if in mixed-precision mode
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_score_norm": scores_normalized}, "SAVE_TENSOR")

    # dropping out the attention links for each of the heads
    # this is different from what we usually do
    if is_training:
      # scores_normalized = tf.layers.dropout(
      #     inputs=scores_normalized, 
      #     rate=dropout_rate,
      #     training=is_training)
      scores_normalized = common_layers.npu_dropout(
          input_tensor=scores_normalized,
          dropout_prob=dropout_rate)
    # v shape: [batch, num_heads, mem_length, units]
    # score shape: [batch, num_heads, query_length, memory_length]
    # context shape: [batch, num_heads, query_length, units]
    context = tf.matmul(scores_normalized, v)
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_context": context}, "SAVE_TENSOR")

    if edge_v is not None:
      context += _relative_attention_inner(scores_normalized, edge_v, False)
    return scores_normalized, context


def additive_attention(q,
                        k,
                        v,
                        mask,
                        dropout_rate,
                        is_training,
                        name=None):
    """additive attention.

    Args:
      q: a Tensor with shape [batch, heads, length_q, depth_k]
      k: a Tensor with shape [batch, heads, length_kv, depth_k]
      v: a Tensor with shape [batch, heads, length_kv, depth_v]
      bias: bias Tensor (see attention_bias())
      dropout_rate: a floating point number
      name: an optional string

    Returns:
      A Tensor.
    """
    with tf.variable_scope(
            name, default_name="additive_attention", values=[q, k, v]) as scope:
        _units = q.get_shape().as_list()[-1]
        # init vars in additive attention
        q_att = tf.layers.Dense(
            units=_units,
            use_bias=False,
            name="query_att"
        )
        k_att = tf.layers.Dense(
            units=_units,
            use_bias=False,
            name="key_att"
        )
        v_att = tf.layers.Dense(
            units=1,
            use_bias=False,
            name="value_att"
        )

        # key shape: [batch, num_heads, mem_length, units]
        keys = k_att(k)
        # reshape to [batch, num_heads, 1, mem_length, units]
        keys = tf.expand_dims(keys, axis=2)
        # query shape: [batch, num_heads, query_length, units]
        query = q_att(q)
        # reshape to [batch, num_heads, query_length, 1, units]
        query = tf.expand_dims(query, axis=3)
        # scores shape: [batch, num_heads, query_length, memory_length]
        scores = tf.squeeze(v_att(tf.tanh(query + keys)), axis=-1)

        # mask out paddings
        if mask is not None:
            # mask shape [batch, query_length, mem_length]
            mshape = tf.shape(mask)
            # assert len(mask.get_shape().as_list()) == 3
            mask = tf.expand_dims(mask, 1)
            if len(mask.get_shape().as_list()) == 3:
                mask = tf.expand_dims(mask, 1)

            scores = scores * mask + ((1.0 - mask) * constant_utils.DT_FLOAT().min)
        scores_normalized = tf.nn.softmax(scores, name="score_norm")
        # dropping out the attention links for each of the heads
        # this is different from what we usually do
        if is_training:
            # scores_normalized = tf.layers.dropout(
            #     inputs=scores_normalized,
            #     rate=dropout_rate,
            #     training=is_training)
            scores_normalized = common_layers.npu_dropout(
                inputs=scores_normalized,
                dropout_prob=dropout_rate)
        # v shape: [batch, num_heads, mem_length, units]
        # score shape: [batch, num_heads, query_length, memory_length]
        # context shape: [batch, num_heads, query_length, units]
        context = tf.matmul(scores_normalized, v)

        return scores_normalized, context

def multihead_attention(query,
                        memory,
                        mask,
                        key_depth,
                        value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        is_training,
                        name=None,
                        cache=None,
                        branch=False,
                        filter_depth=None,
                        activation="relu",
                        relpos=0,
                        sum_att=False,
                        batch_size=None,
                        from_seq_len=None,
                        to_seq_len=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string.
    **kwargs (dict): Parameters for the attention function

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  # if batch_size is not None:
  #   query = tf.reshape(query, [-1, key_depth])

  if key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (key_depth, num_heads))
  if value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (value_depth, num_heads))
  with tf.variable_scope("multihead_attention",
      values=[query, memory], reuse=tf.AUTO_REUSE):
    
    if memory is not None and cache is not None:
      k = cache["k_encdec"]
      v = cache["v_encdec"]

      

      q = compute_qkv_fused(query, memory, key_depth, value_depth, ignore_kv=True)
      # reshape to [batch, heads, time, units/heads]
      
      q = tf.reshape(q, [batch_size, from_seq_len, num_heads, key_depth//num_heads])
      q = tf.transpose(q, [0, 2, 1, 3])

      graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_key_dense": k}, "SAVE_TENSOR")
      graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_value_dense": v}, "SAVE_TENSOR")
      
    else:
      # project query and memory
      #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_query": query}, "SAVE_TENSOR")
      q, k, v = compute_qkv_fused(query, memory, key_depth, value_depth)
	  
      # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_qkv": q}, "SAVE_TENSOR")

      # if batch_size is None:
      #   # reshape to [batch, heads, time, units/heads]
      #   q = split_heads(q, num_heads)
      #   #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_q2": q}, "SAVE_TENSOR")
      #   k = split_heads(k, num_heads)
      #   v = split_heads(v, num_heads)
      # else:
      q = tf.reshape(q, [batch_size, from_seq_len, num_heads, key_depth//num_heads])
      q = tf.transpose(q, [0, 2, 1, 3])
      k = tf.reshape(k, [batch_size, to_seq_len, num_heads, key_depth//num_heads])
      k = tf.transpose(k, [0, 2, 1, 3])
      v = tf.reshape(v, [batch_size, to_seq_len, num_heads, value_depth//num_heads])
      v = tf.transpose(v, [0, 2, 1, 3])

      if cache is not None:
        k = cache["k"] = tf.concat([cache["k"], k], axis=2)
        v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        k = k[:,:,1:]
        v = v[:,:,1:]

    # scale query
    key_depth_per_head = key_depth // num_heads
    q *= key_depth_per_head**-0.5


    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_q3": q}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_k": k}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_v": v}, "SAVE_TENSOR")

    if relpos > 0: 
      """Implement relative position representation 
        https://arxiv.org/pdf/1803.02155.pdf"""
      window_size = 2 * relpos + 1
      with tf.variable_scope("relpos"):
        weight_k = tf.get_variable(
            "k", [window_size, key_depth // num_heads], dtype=constant_utils.DT_FLOAT())
        weight_v = tf.get_variable(
            "v", [window_size, value_depth // num_heads], dtype=constant_utils.DT_FLOAT())
        
        # qidx = tf.range(tf.shape(q)[2])
        length = tf.shape(k)[2]
        indices = tf.range(length)
        # shape [length, length]
        # [[0,-1,-2],
        #  [1, 0,-1],
        #  [2, 1, 0],
        # ]
        # during inference take the last query
        qindices = indices
        if cache is not None:
          qindices = indices[-1:]
          
        indices = tf.expand_dims(qindices, 1) - tf.expand_dims(indices, 0)
        
        # clip to [-relpos, relpos]
        # then shift to [0, 2*relpos]
        indices = tf.maximum(tf.minimum(indices, relpos), -relpos) + relpos

        # use indices to lookup edge reprentation
        # shape [length, length, unit]
        edge_k = tf.nn.embedding_lookup(weight_k, indices)
        edge_v = tf.nn.embedding_lookup(weight_v, indices)
    else:
      edge_k = None
      edge_v = None

    if sum_att:
        scores, x = additive_attention(
            q, k, v, mask, dropout_rate, is_training=is_training)
    else:
        scores, x = dot_product_attention(
            q, k, v, mask, dropout_rate, is_training=is_training,
            edge_k=edge_k, edge_v=edge_v)

    # x shape: [batch, num_heads, query_length, units/num_heads]
    # combined to [batch, query_length, units]
    def combine_heads(x):
      """Reshape x so that the last two dimension become one.
      Args:
        x: a Tensor with shape [..., a, b]
      Returns:
        a Tensor with shape [..., ab]
      """
      x_shape = shape_list(x)
      a, b = x_shape[-2:]
      return tf.reshape(x, x_shape[:-2] + [a * b])
    
    if branch:
      # weighted transformer
      with tf.variable_scope("branch"):
        branch_k = tf.get_variable("k", [num_heads], dtype=constant_utils.DT_FLOAT())
        branch_alpha = tf.get_variable("alpha", [num_heads], dtype=constant_utils.DT_FLOAT())
        # [batch, heads, q_len, unit]
        x = tf.layers.dense(
            x, output_depth, use_bias=False, name="o")
        x = x * tf.reshape(
                  tf.nn.softmax(branch_k), 
                  [1, num_heads, 1, 1])
        x = position_feed_forward(
                x = x, 
                filter_depth=filter_depth, 
                output_depth=output_depth, 
                dropout_rate=dropout_rate, 
                is_training=is_training,
                activation=activation)
        # batch x q_len x units
        x = tf.reduce_sum(
                x * tf.reshape(
                        tf.nn.softmax(branch_alpha), 
                        [1, num_heads, 1, 1]),
                axis=1)
        
    else:
      # if batch_size is None:
      #   x = combine_heads(tf.transpose(x, [0,2,1,3]))
      # else:
      x = tf.transpose(x, [0, 2, 1, 3])
      x = tf.reshape(x, [-1, value_depth])

      # after combine, linear proj to output_depth
      x = tf.layers.dense(
          x, output_depth, use_bias=False, name="output_transform")
      
      graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_output": x}, "SAVE_TENSOR")

    # scores shape: [batch, num_heads, query_length, mem_length]
    # we take the average over num_heads
    # out shape: [batch, query_length, mem_length]
    # scores = tf.reduce_mean(scores, axis=1)
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_after_self_att": x}, "SAVE_TENSOR")
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_after_self_att": scores}, "SAVE_TENSOR")
    return scores, x


def position_feed_forward(x, filter_depth, output_depth, dropout_rate, is_training, activation="relu"):
  """ position-wise feed forward: dense-relu-dense
  Args:
    x: input tensor with shape [batch, length, units]
    mask: input mask with shape [batch length]
    filter_depth: the num of units of hidden layer
    output_depth: the number of units of output layer
    dropout_rate:
    is_training
  Return:
    a tensor with shape [batch, length, output_depth]
  """
  if activation == "relu":
    act_fn = tf.nn.relu
  elif activation == "swish":
    act_fn = activation_swish
  else:
    raise ValueError("Unknown activation")
  
  x_shape = common_layers.shape_list(x)
  x=tf.reshape(x, [-1, x_shape[-1]])

  # one layer ff with relu
  h = tf.layers.dense(
      x, filter_depth, use_bias=True, activation=act_fn, name="conv1")
  # dropout
  if is_training:
    # h = tf.layers.dropout(
    #     inputs=h, 
    #     rate=dropout_rate,
    #     training=is_training)
    h = common_layers.npu_dropout(
        input_tensor=h, 
        dropout_prob=dropout_rate)

  # another ff
  o = tf.layers.dense(h, output_depth, use_bias=True, name="conv2")

  o = tf.reshape(o, x_shape[:-1] + [output_depth])
  return o


def self_attention_sublayer(x, mask, num_units, num_heads, dropout_rate, is_training, memory=None, cache=None, branch=False, 
                            filter_depth=None, activation="relu", relpos=0, sum_att=False,
                            batch_size=None, from_seq_len=None, to_seq_len=None):
  # with tf.variable_scope("self_attention"):
  #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/x": x}, "SAVE_TENSOR")
  att_scores, x = multihead_attention(
      query=x,
      memory=memory, # set none to use query as memory
      mask=mask,
      key_depth=num_units,
      value_depth=num_units,
      output_depth=num_units,
      num_heads=num_heads,
      dropout_rate=dropout_rate,
      is_training=is_training,
      cache=cache,
      branch=branch,
      filter_depth=filter_depth,
      activation=activation,
      relpos=relpos,
      sum_att=sum_att,
      batch_size=batch_size,
      from_seq_len=from_seq_len,
      to_seq_len=to_seq_len,
  )
  return att_scores, x



def average_attention_sublayer(x, mask, num_units, dropout_rate, 
                               is_training, cache=None, 
                               filter_depth=None, activation="relu", position=None):
  y = x
  # accumulate
  if cache is None:
    # x shape: batch x length x units
    # mask shape: batch x length x length, triangle
    length = tf.shape(y)[1]
    accum_mask = tf.cast(mask, constant_utils.DT_FLOAT()) / tf.reshape(
          tf.cast(tf.range(length) + 1, constant_utils.DT_FLOAT()), 
          [1, length, 1])
    # shape: batch x length x unit
    accum_y = tf.matmul(accum_mask, y)
  else:
    assert mask is None
    position = tf.cast(position, constant_utils.DT_FLOAT())
    cache_len = tf.cast(tf.shape(cache["accum_x"])[1], constant_utils.DT_FLOAT())
    accum_y = cache["accum_x"] = (y + cache["accum_x"] * position) / (position + 1.)
  
  with tf.variable_scope("ffn"):
    y_ffn = position_feed_forward(
        x=accum_y,
        filter_depth=filter_depth,
        output_depth=num_units,
        dropout_rate=dropout_rate,
        is_training=is_training,
        activation=activation)

  with tf.variable_scope("gate"):
    input_ = tf.concat([y, y_ffn], axis=-1)
    g_ = tf.layers.dense(
        input_, 2*num_units, use_bias=True, activation=tf.nn.sigmoid)
    # input and forget get
    # batch x length x unit
    gi, gf = tf.split(axis=-1, value=g_, num_or_size_splits=2)
    h = gi * y + gf * y_ffn
  # then layer_norm
#   out = layer_norm(y+h)
  return h


def feed_forward_sublayer(x, filter_depth, num_units, dropout_rate, is_training, cache=None, activation="relu"):
  # the second sublayer is feed-forward
  with tf.variable_scope("ffn"):
    if activation == "relu":
      act_fn = tf.nn.relu
    elif activation == "swish":
      act_fn = activation_swish
    else:
      raise ValueError("Unknown activation")
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_input": x}, "SAVE_TENSOR")

    # one layer ff with relu
    h = tf.layers.dense(
        x, filter_depth, use_bias=True, activation=act_fn, name="conv1")
    
    graph_utils.add_dict_to_collection({tf.get_variable_scope().name+"_hidden": h}, "SAVE_TENSOR")

    # dropout
    if is_training:
      # h = tf.layers.dropout(
      #     inputs=h, 
      #     rate=dropout_rate,
      #     training=is_training)
      h = common_layers.npu_dropout(
          input_tensor=h,
          dropout_prob=dropout_rate)

    # another ff
    x = tf.layers.dense(h, num_units, use_bias=True, name="conv2")

    graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_output": x}, "SAVE_TENSOR")
    
    return x


def layer_process(x, y=None, mode=None, dropout_rate=0, is_training=False):
  if not mode or mode == "none":
    return x
  
  if not is_training:
    dropout_rate = 0

  #graph_utils.add_dict_to_collection({"encoder_emb_processed2": x}, "SAVE_TENSOR")
  #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/encoder_emb_processed23": x}, "SAVE_TENSOR")
  for m in mode:
    if m == 'a':
      x += y
      #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_after_reslink": x}, "SAVE_TENSOR")
    elif m == 'n':
      x = layer_norm(x)
    elif m == 'd':
      # x = tf.layers.dropout(
      #     inputs=x, 
      #     rate=dropout_rate,
      #     training=is_training)
      if is_training:
        x = common_layers.npu_dropout(
            input_tensor=x,
            dropout_prob=dropout_rate)
    else:
      raise ValueError("Unknown layer_process mode")
  #graph_utils.add_dict_to_collection({"encoder_emb_processed3": x}, "SAVE_TENSOR")
  return x
