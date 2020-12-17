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

"""beam search utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import tensor_util

from noahnmt.utils import constant_utils
from noahnmt.layers import common_layers as common_utils


class BeamSearchConfig(
    namedtuple("BeamSearchConfig", [
        "beam_width", "vocab_size", "eos_token", "length_penalty_weight",
        "coverage_penalty_weight", "stop_early",
        "keep_finished", "return_top_beam"
    ])):
  """Configuration object for beam search.

  Args:
    beam_width: Number of beams to use, an integer
    vocab_size: Output vocabulary size
    eos_token: The id of the EOS token, used to mark beams as "done"
    length_penalty_weight: Weight for the length penalty factor. 0.0 disables
      the penalty.
    choose_successors_fn: A function used to choose beam successors based
      on their scores. Maps from (scores, config) => (chosen scores, chosen_ids)
    keep_finished: once find a finished hypo, keep it always in beams
  """
  pass


class BeamSearchStepOutput(
    namedtuple("BeamSearchStepOutput",
               ["predicted_ids", "beam_parent_ids"])):
  """Outputs for a single step of beam search.
  Args:
    scores: Score for each beam, a float32 vector
    predicted_ids: predictions for this step step, an int32 vector
    beam_parent_ids: an int32 vector containing the beam indices of the
      continued beams from the previous step
  """
  pass

class BeamSearchBase(object):
  def __init__(self, batch_size, config, num_beams=None):
    self.batch_size = batch_size
    self.config = config
    self.beam_width = config.beam_width

    if num_beams is None:
      self.num_beams = batch_size * self.beam_width
    else:
      self.num_beams = num_beams
    
    self.batch_indices = _compute_batch_indices(
        self.batch_size, self.beam_width, self.num_beams)


def _tile_batch(tensor, beam_size):
  tensor = _expand_to_beam_size(tensor, beam_size)
  return _merge_beam_dim(tensor)

def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)


def _merge_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.

  Args:
    tensor: Tensor to reshape of shape [A, B, ...]

  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = common_utils.shape_list(tensor)
  shape[0] *= shape[1]  # batch -> batch * beam_size
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].

  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.

  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = common_utils.shape_list(tensor)
  new_shape = [batch_size] + [beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _log_prob_from_logits(logits):
  with tf.name_scope("log_prob"):
    return tf.nn.log_softmax(logits)
    #return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def _compute_batch_indices(batch_size, beam_width, num_beams=None):
  """Computes the i'th coodinate that contains the batch index for gathers.

  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.

  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_width] tensor of ids
  """
  # The next three steps are to create coordinates for tf.gather_nd to pull
  # out the topk sequences from sequences based on scores.
  # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  # batch the beam item is in. This will create the i of the i,j coordinate
  # needed for the gather
  if num_beams is None:
    num_beams = batch_size * beam_width
  batch_pos = tf.range(num_beams) // beam_width
  batch_pos = tf.reshape(batch_pos, [batch_size, beam_width])
  return batch_pos


# def _calc_offset(batch_size, beam_width):
#   offset = tf.range(batch_size) * beam_width
#   offset = tf.tile(tf.expand_dims(offset, 1), [1, beam_width])
#   return tf.reshape(offset, [-1])


def _calc_gathernd_corrdinates(batch_indices, topk_indices):
  # The next three steps are to create coordinates for tf.gather_nd to pull
  # out the topk sequences from sequences based on scores.
  # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  # batch the beam item is in. This will create the i of the i,j coordinate
  # needed for the gather
  # batch_pos = _compute_batch_indices(batch_size, beam_width, num_beams)
  # top coordinates will give us the actual coordinates to do the gather.
  # stacking will create a tensor of dimension batch * beam * 2, where the
  # last dimension contains the i,j gathering coordinates.
  top_coordinates = tf.stack([batch_indices, topk_indices], axis=2)
  return top_coordinates


def _choose_top_k(scores, beam_width, batch_indices, num_beams=None, prefix="default"):
  topk_scores, topk_indexes = tf.nn.top_k(scores, k=beam_width)
  top_coordinates = _calc_gathernd_corrdinates(batch_indices, topk_indexes)
  # Gather up the highest scoring sequences.  For each operation added, give it
  # a concrete name to simplify observing these operations with tfdbg.  Clients
  # can capture these tensors by watching these node names.
  def gather(tensor):
    return tf.gather_nd(tensor, top_coordinates, name=prefix)

  return topk_scores, gather


def _calc_length_penalty(length, alpha):
  return tf.pow(((5. + tf.cast(length, constant_utils.DT_FLOAT())) / 6.), alpha)


def _calc_coverage_penalty(attention_scores, beta, sequence_length=None):
  # input shape: batch x beam x tgt_len x src_len
  assert attention_scores.get_shape().ndims > 3
  if sequence_length is not None:
    mask = tf.sequence_mask(
        sequence_length, 
        tf.shape(attention_scores)[1], 
        dtype=constant_utils.DT_FLOAT())
    attention_scores = attention_scores * tf.expand_dims(mask, -1)

  # batch x beam x source_len
  alignments = tf.minimum(
      tf.reduce_sum(attention_scores, axis=2), 1.0)
  # zeors to ones for tf.log
  alignments = tf.where(alignments>0,
                        alignments,
                        tf.ones_like(alignments))
  return beta * tf.reduce_sum(tf.log(alignments), axis=2)


def _mask_finished_probs(probs, eos_token, finished):
  """Masks log probabilities such that finished beams
  allocate all probability mass to eos. Unfinished beams remain unchanged.

  Args:
    probs: Log probabiltiies of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to
    finished: A boolean tensor of shape `[batch_size, beam_width]` that specifies which
      elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished beams
    stay unchanged and finished beams are replaced with a tensor that has all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[2]
  finished_mask = tf.expand_dims(tf.to_float(1. - tf.to_float(finished)), 2)
  # These examples are not finished and we leave them
  non_finished_examples = finished_mask * probs
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=constant_utils.DT_FLOAT(),
      on_value=0.,
      off_value=constant_utils.DT_FLOAT().min)
  finished_examples = (1. - finished_mask) * finished_row
  return finished_examples + non_finished_examples

def _get_state_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  if isinstance(tensor, tf.Tensor):
    shape = tensor.get_shape().as_list()
    if len(shape) < 3:
      return tensor.get_shape()
    if len(shape) > 3:
      for i in range(1, len(shape)-1):
        shape[i] = None
      return tf.TensorShape(shape)
    return tf.TensorShape([None]*len(shape))
  elif isinstance(tensor, tf.TensorArray):
    return tf.TensorShape(None)
  else:
    raise ValueError("not supported")

def get_shape_invariants(states):
  return nest.map_structure(_get_state_shape_invariants, states)
