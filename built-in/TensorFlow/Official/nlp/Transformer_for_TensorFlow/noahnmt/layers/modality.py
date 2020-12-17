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

"""Modality base class - defines the bottom and top of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from noahnmt.layers import common_layers
from noahnmt.utils import registry
from noahnmt.utils import vocab_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import graph_utils


class SymbolModality(object):
  """Modality class for data transformations.

  An abstract class representing modalities for transforming data to a space
  interpretable by T2T models. It has 4 functions:
  * bottom: called on inputs entering the model.
  * targets_bottom: called on targets entering the model (e.g., the decoder).
  * top: called on model outputs to generate predictions (e.g., logits).
  * loss: called on predictions (outputs of top) and targets.

  For example, think about a modality for images:
  * `bottom` represents the part of the model applied to an incoming image,
    e.g., an entry flow of a convolutional network.
  * `top` represents the top part of a model that is generating images, e.g., a
    PixelCNN network.
  * `targets_bottom` represents the auto-regressive part of the network.  It is
    applied to the already-generated part of an image, which is given to the
    decoder to generate the next part. In some cases, e.g., for text, it is the
    same as the `bottom` function, and that is the default we use. But, e.g.,
    for images, a different function might be needed to regress properly.
  * `loss` would compare the generated image to the target image and score it.

  All the functions have simple and sharded versions. A sub-class only needs to
  implement the simple version, the default sharding will be used then.
  """

  def __init__(self, 
               vocab_info, 
               num_units, 
               weight_tying=False, 
               partitioner=None,
               embedding_multiply_mode="",
               name="symbol_modality",
               mos_n_experts=0,
               embedding_initializer=None,
               softmax_bias = False,
               **kwargs):
    """ init
    """
    self.name = name
    self.vocab_info = vocab_info
    self.num_units = num_units
    self.partitioner = partitioner
    self.weight_tying = weight_tying
    self.embedding_multiply_mode = embedding_multiply_mode
    self.selected_vocab_ids = None
    self.mos_n_experts = mos_n_experts
    self.embedding_initializer = embedding_initializer
    self.softmax_bias = softmax_bias

    self._bottom_built = False
    self._top_built = False


  def set_selected_vocab(self, vocab_ids):
    """ selected vocab is enabled
    Call this function before calling other functions
    """
    self.selected_vocab_ids = vocab_ids


  @property
  def vocab_size(self):
    return self.vocab_info.vocab_size
  
  @property
  def eos(self):
    return self.vocab_info.special_vocab.eos
  
  @property
  def sos(self):
    return self.vocab_info.special_vocab.sos
    

  def _get_initializer(self, hidden_dim=None):
    if self.embedding_initializer:
      if self.embedding_initializer == "normal":
        return tf.random_normal_initializer(0.0, self.num_units**-0.5)
      else:
        raise ValueError("Unknown embedding initializer")
    else:
      return None

  def _build_bottom(self):
    if self._bottom_built:
      return

    self._bottom_built = True
    with tf.variable_scope(self.name):
      self.bottom_weight = tf.get_variable(
          name="embedding",
          shape=[self.vocab_size, self.num_units],
          initializer=self._get_initializer(),
          partitioner=self.partitioner,
          dtype=tf.float32)
    
    # use selected vocab
    # target_bottom_weight is created
    # in case source and target modalities are shared
    self.target_bottom_weight = self.bottom_weight
    if self.selected_vocab_ids:
      self.target_bottom_weight = tf.gather(
          self.bottom_weight, self.selected_vocab_ids)

  
  def _build_top(self, hidden_dim=None):
    if self._top_built:
      return
    self._top_built = True
    
    if not hidden_dim:
      hidden_dim = self.num_units
    
    # mixture of softmax
    if self.mos_n_experts > 1:
      with tf.variable_scope(self.name):
        self.mos_prior = tf.get_variable(
            name="mos_prior",
            shape=[hidden_dim, self.mos_n_experts],
            dtype=constant_utils.DT_FLOAT())
        self.mos_latent = tf.get_variable(
            name="mos_latent",
            shape=[hidden_dim, self.mos_n_experts * hidden_dim],
            dtype=constant_utils.DT_FLOAT())

    if self.softmax_bias:
      self.output_bias = tf.get_variable(
            name="output_bias",
            shape=[self.vocab_size],
            initializer=self._get_initializer(),
            dtype=constant_utils.DT_FLOAT())

    if self.weight_tying:
      self.top_weight = self.target_bottom_weight
    else:
      with tf.variable_scope(self.name):
        self.top_weight = tf.get_variable(
            name="softmax",
            shape=[self.vocab_size, hidden_dim],
            initializer=self._get_initializer(),
            partitioner=self.partitioner,
            dtype=constant_utils.DT_FLOAT())
      
      # selected vocab
      if self.selected_vocab_ids:
        self.top_weight = tf.gather(
            self.top_weight, self.selected_vocab_ids)     


  def bottom(self, tensor):
    """Transform one shard of input.

    Args:
      tensor: An int32 Tensor with shape [batch, length]
    Returns:
      A float32 Tensor with shape [batch, length, body_input_depth]
    """
    if not self._bottom_built:
      self._build_bottom()
      
    shape_list = common_layers.shape_list(tensor)
    flat_tensor = tf.reshape(tensor, [-1])
    token_type_embeddings = common_layers.gather_npu(self.bottom_weight, flat_tensor)
    tensor_embeded = tf.reshape(token_type_embeddings, shape_list + [self.num_units])

    if self.embedding_multiply_mode == "sqrt_depth":
      tensor_embeded *= self.num_units**0.5
    elif self.embedding_multiply_mode == "rsqrt_depth":
      tensor_embeded /= self.num_units**0.5
    return tensor_embeded


  # def bottom_sharded(self, xs, data_parallelism):
  #   """Transform the inputs.

  #   Args:
  #     xs: A list of num_datashards Tensors (one per shard)
  #       each with shape [batch, p0, p1, depth]
  #     data_parallelism: a expert_utils.Parallelism object
  #   Returns:
  #     shaded_body_input: A list of num_datashards Tensors, each with shape
  #       [batch, p0, p1, body_input_depth].
  #   """
  #   return data_parallelism(self.bottom, xs)


  def target_bottom(self, tensor):
    """Transform one shard of targets.

    Args:
      tensor: An int32 Tensor with shape [batch, length]
    Returns:
      A float32 Tensor with shape [batch, length, body_input_depth]
    """
    if not self._bottom_built:
      self._build_bottom()

    # tensor_embeded = tf.nn.embedding_lookup(
    #     self.target_bottom_weight, tensor)
    shape_list = common_layers.shape_list(tensor)
    flat_tensor = tf.reshape(tensor, [-1])
    tensor_embeded = common_layers.gather_npu(self.target_bottom_weight, flat_tensor)
    tensor_embeded = tf.reshape(tensor_embeded, shape_list + [self.num_units])

    if self.embedding_multiply_mode == "sqrt_depth":
      tensor_embeded *= self.num_units**0.5
    elif self.embedding_multiply_mode == "rsqrt_depth":
      tensor_embeded /= self.num_units**0.5
    return tensor_embeded


  # def target_bottom_sharded(self, xs, data_parallelism):
  #   """Transform the target.

  #   Args:
  #     xs: A list of num_datashards Tensors (one per shard)
  #       each with shape [batch, p0, p1, depth]
  #     data_parallelism: a expert_utils.Parallelism object
  #   Returns:
  #     shaded_body_input: A list of num_datashards Tensors, each with shape
  #       [batch, p0, p1, body_input_depth].
  #   """
  #   return data_parallelism(self.target_bottom, xs)


  def top(self, body_output):
    """Generate predictions/logits for one shard of output.

    Most classes will override this function.

    Args:
      body_output: A Tensor with shape [batch, p0, body_output_depth]
    Returns:
      A Tensor of class logits.
    """
    body_output_shape = common_layers.shape_list(body_output)
    hidden_dim = body_output_shape[-1]
    if not self._top_built:
      self._build_top(hidden_dim)
    
    # reshape to 2-D
    body_output = tf.reshape(body_output, [-1, hidden_dim])

    # mixture of softmax
    if self.mos_n_experts > 1:
      prior = tf.nn.softmax(tf.matmul(body_output, self.mos_prior))
      # [batch*experts, units]
      body_output = tf.reshape(
          tf.matmul(body_output, self.mos_latent),
          [-1, hidden_dim])

    # matmul
    # body_output = tf.cast(body_output, tf.float32)
    top_weight = tf.cast(self.top_weight, body_output.dtype)
    logits = tf.matmul(body_output, top_weight, transpose_b=True)
    graph_utils.add_dict_to_collection({"logits": logits}, "SAVE_TENSOR")
    # logits = tf.cast(logits, tf.float16)


    # mixture of softmax
    if self.mos_n_experts > 1:
      logits = tf.reshape(logits, [-1, self.mos_n_experts, self.vocab_size])
      logits = tf.reduce_sum(tf.expand_dims(prior, -1) * logits, 1)

    # reshape back
    logits = tf.reshape(logits, body_output_shape[:-1] + [self.vocab_size])

    if self.softmax_bias:
      logits = tf.nn.bias_add(logits, self.output_bias)
    
    return logits


  # def top_sharded(self, sharded_body_output, sharded_targets, data_parallelism):
  #   """Generate predictions/logits for all shards.

  #   Classes with cross-shard interaction will override this function.

  #   Args:
  #     sharded_body_output: A list of Tensors.
  #     sharded_targets: A list of Tensors.
  #     data_parallelism: a expert_utils.Parallelism object.
  #   Returns:
  #     sharded_logits: A list of Tensors.
  #   """
  #   return data_parallelism(self.top, sharded_body_output, sharded_targets)


  # def loss(self, logits, targets, target_length=None, label_smoothing=None):
  #   """Compute loss numerator and denominator for one shard of output."""
  #   return loss_utils.compute_nce_loss(
  #       logits,
  #       targets,
  #       target_length,
  #       label_smoothing)


  # def loss_sharded(self, sharded_logits, sharded_targets, data_parallelism, sharded_target_length=None):
  #   """Compute loss for all shards."""
  #   sharded_loss_num, sharded_loss_den = data_parallelism(
  #       self.loss, sharded_logits, sharded_targets)
  #   loss = tf.add_n(sharded_loss_num) / tf.maximum(1.0,
  #                                                  tf.add_n(sharded_loss_den))
  #   return loss
