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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys
import six
import math
import numpy as np
import scipy
import codecs

import tensorflow as tf

from noahnmt.utils import constant_utils


def transform_align_prob(align_dict):
  """
  transformation on align probs:
    2. normalize many-to-one
  """

  # normalize many to one
  for t, s_prob in align_dict.items():
    if len(s_prob) > 1:
      for s in s_prob:
        align_dict[t][s] = 1.0 / len(s_prob)
  
  return align_dict


def create_sparse_tensor(align_pairs):
  # create align dict: t:s:prob
  align_dict = {}
  for s, t, p in align_pairs:
    if not t in align_dict:
      align_dict[t] = {}
    
    assert not s in align_dict[t]
    align_dict[t][s] = p

  # translation align probabilities
  if len(align_dict) == 0:
    align_dict[-1] = {-1: 1.0}
  else:
    align_dict = transform_align_prob(align_dict)

  # convenient list for SparseTensorValue
  t_s_prob = []
  for t, s_prob in align_dict.items():
    t_s_prob.extend([(t,s,p) for s, p in s_prob.items()])
  tgt_index, src_index, align_probs = zip(*t_s_prob)

  # Wrap `coo_matrix` in the `tf.SparseTensorValue` form that TensorFlow expects.
  # SciPy stores the row and column coordinates as separate vectors, so we must 
  # stack and transpose them to make an indices matrix of the appropriate shape.
  # sparse_value = tf.SparseTensorValue(
  #     indices=np.array([tgt_index, src_index]).T,
  #     values=align_probs,
  #     dense_shape=(max_t, max_s))
  # return tf.SparseTensor.from_value(sparse_value)
  indices=np.array([tgt_index, src_index]).T
  return (indices, align_probs)


def get_align_dataset(align_file):
  """
  a dataset containing alignment SparseTensorValues

  Args:
    align_file: file path which contains alignmets: 0-1 1-1 ...
  
  Returns:
    a tf.data.Dataset object
  """

  def generator():
    if tf.gfile.Exists(align_file):
      vocab2id = {}
      with codecs.getreader("utf-8")(tf.gfile.GFile(align_file, "rb")) as f:
        vocab_size = 0
        for line in f:
          align_pairs = [t.split('-') for t in line.strip().split()]
          if len(align_pairs[0]) < 3:
            align_pairs = [(s, t, 1) for s, t in align_pairs]
          # add 1 so that paddings can use 0
          align_pairs = [(int(s)+1, int(t)+1, float(p)) for s, t, p in align_pairs]
          # sparse_tensor = create_sparse_tensor(align_pairs)
          yield create_sparse_tensor(align_pairs)
    else:
      raise ValueError("align_file does not exist.")

  dataset = tf.data.Dataset.from_generator(
      generator, (constant_utils.DT_INT(), 
                  constant_utils.DT_FLOAT()))
  # dataset = dataset.map(
  #     lambda probs, indices, max_t, max_s: 
  #       tf.sparse_tensor_to_dense(tf.SparseTensor(indices, probs, (max_t, max_s))))
  return dataset


def create_dense_matrix(features):
  """
  create a dense alignment matrix

  Args:
    features: dict of tensors, with keys:
      align_indices: shape (batch x len x 2), int64, sparse indices
                batch:len:0 --> target index
                batch:len:1 --> source index
      align_values: shape (batch x len), float, alignment probs (default 1.0)
  
  Returns:
    align_matrix: batch x tgt_len x src_len
  """
  batch_size = tf.shape(features["source_tokens"])[0]
  max_src_len = tf.shape(features["source_tokens"])[1]
  # -1 because we add both <s> and </s> to target tokens
  max_tgt_len = tf.shape(features["target_tokens"])[1] - 1
  num_indices = tf.shape(features["align_indices"])[1]

  # create batch_pos for creating indices (batch_pos, tgt_pos, src_pos)
  batch_pos = tf.range(batch_size * num_indices, dtype=constant_utils.DT_INT())
  batch_pos = tf.floordiv(batch_pos, num_indices)
  # real indices with batch_pos: [batch x len, 3]
  indices = tf.concat([
        tf.expand_dims(batch_pos, 1),
        tf.reshape(features["align_indices"], 
                   [batch_size * num_indices, 2])],
      axis=1)
  
  # create dense tensor: batch x tgt_len x src_len
  # +1 because pos 0 is for paddings
  dense_shape = tf.to_int64(
      tf.stack([batch_size, max_tgt_len+1, max_src_len+1], 
                axis=0))
  values = tf.reshape(features["align_values"], [-1])
  align_matrix = tf.sparse_to_dense(
      sparse_indices=tf.to_int64(indices), 
      sparse_values=values, 
      output_shape=dense_shape,
      validate_indices=False)
  align_matrix = align_matrix[:, 1:, 1:]
  
  # padding mask
  # batch x src_len
  src_mask = tf.sequence_mask(
      features["source_len"], 
      maxlen=max_src_len, 
      dtype=constant_utils.DT_FLOAT())
  # batch x tgt_len
  tgt_mask = tf.sequence_mask(
      features["target_len"], 
      maxlen=max_tgt_len, 
      dtype=constant_utils.DT_FLOAT())
  # batch x tgt_len x src_len
  align_mask = tf.matmul(
      tf.expand_dims(tgt_mask, 2), 
      tf.expand_dims(src_mask, 1))

  return align_matrix, align_mask


def weight_decay(
            weight,
            start_decay_step,
            decay_steps,
            decay_factor,
            stop_decay_at):
  weight = tf.constant(weight)
  global_step = tf.train.get_global_step()

  decay = tf.train.exponential_decay(
      weight,
      (global_step - start_decay_step),
      decay_steps, decay_factor, staircase=True)

  decay_weight = tf.cond(
      global_step < start_decay_step,
      lambda: weight,
      lambda: decay,
      name="guided_attention_decay_cond")
  
  decay_weight = tf.cond(
      global_step > stop_decay_at,
      lambda: 0.,
      lambda: decay_weight)
  return decay_weight


# if __name__ == "__main__":
#   import tempfile
#   alignments = ["0-0 0-1 1-1 2-3"] * 10
#   align_file = tempfile.NamedTemporaryFile()
#   align_file.write("\n".join(alignments).encode("utf-8"))
#   align_file.flush()

#   def batching_func(batch_size, x):
#     padded_shapes = (tf.TensorShape([2, None]), 
#                      tf.TensorShape([None])) # align
#     padded_values = (-1, 0.)
#     return x.padded_batch(
#         batch_size,
#         padded_shapes=padded_shapes,
#         padding_values=padded_values)

#   dataset = get_align_dataset(align_file.name)
#   dataset = dataset.repeat(-1)
#   dataset = batching_func(3, dataset)
#   iterator = dataset.make_one_shot_iterator()
#   indices, values = iterator.get_next()
#   # dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor)

#   with tf.Session() as sess:
#     for i in range(15):
#       print(sess.run(indices).shape)