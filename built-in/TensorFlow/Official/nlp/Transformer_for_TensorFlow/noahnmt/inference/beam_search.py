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

"""In-Graph Beam Search Implementation.
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
from noahnmt.inference import beam_search_utils as bs_utils

INF = 1. * 1e9

class BeamSearchState(
    namedtuple("BeamSearchState", 
        ["sequences", "log_probs", "attention_scores", "lengths",
         "finished"])):
  """State for a single step of beam search.

  Args:
    log_probs: The current log probabilities of all beams
    finished: A boolean vector that specifies which beams are finished
    lengths: Lengths of all beams
  """
  pass


class BeamSearch(bs_utils.BeamSearchBase):
  def __init__(self, batch_size, config, num_beams=None):
    super(BeamSearch, self).__init__(
        batch_size, config, num_beams)
    
    # [[0,1,2],[0,1,2]]
    self.init_beam_ids = tf.tile(
        tf.expand_dims(
            tf.range(self.config.beam_width), 0), 
        [batch_size, 1])
    

  def create_initial_beam_state(self, attention_length, initial_ids):
    """Creates an instance of `BeamState` that can be used on the first
    call to `beam_step`.

    Args:
      config: A BeamSearchConfig

    Returns:
      An instance of `BeamState`.
    """
    batch_size = self.batch_size
    beam_width = self.beam_width

    assert attention_length is not None
    attention_scores = tf.zeros(
        [batch_size, beam_width, 1, attention_length],
        dtype=constant_utils.DT_FLOAT())
    
    # init_seq = tf.zeros([batch_size, beam_width, 0])
    init_seq = tf.reshape(initial_ids, [batch_size, beam_width, 1])

    initial_log_probs = tf.constant([[0.] + [-INF] * (beam_width - 1)], dtype=constant_utils.DT_FLOAT())
    initial_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

    return BeamSearchState(
        sequences=init_seq,
        log_probs=initial_log_probs,
        attention_scores=attention_scores,
        finished=tf.zeros(
            [batch_size, beam_width], dtype=tf.bool),
        lengths=tf.zeros(
            [batch_size, beam_width], 
            dtype=constant_utils.DT_INT())
    )


  def update_final_state(self, final_beam_state):
    beta = self.config.coverage_penalty_weight
    alpha = self.config.length_penalty_weight
    beam_width = self.beam_width
    batch_size = self.batch_size

    # only recoring when coverage penalty is used
    # as it is not calculated in the loop on alive_seq
    attention_scores = final_beam_state.attention_scores
    lengths = final_beam_state.lengths
    log_probs = final_beam_state.log_probs
    sequences = final_beam_state.sequences

    # rescoring with length and coverage penalty
    rescored = False
    if alpha > 0:
      log_probs = log_probs / bs_utils._calc_length_penalty(lengths, alpha)
      rescored = True
    if beta > 0.:
      log_probs += bs_utils._calc_coverage_penalty(
          attention_scores, beta, sequence_length=lengths)
      rescored = True
    
    # if rescores, sort hypos
    # if rescored:
    if self.config.keep_finished:
      log_probs, gather_fn = bs_utils._choose_top_k(log_probs, beam_width, self.batch_indices)

      attention_scores = gather_fn(attention_scores)
      # lengths = gather_fn(lengths)
      sequences = gather_fn(sequences)

    # remove <s>
    return sequences[:,:,1:], attention_scores[:,:,1:]
  

  def bound_is_met(self, beam_state, maximum_iterations):
    alpha = self.config.length_penalty_weight
    beta = self.config.coverage_penalty_weight

    finished_mask = tf.cast(beam_state.finished, constant_utils.DT_FLOAT())

    if self.config.keep_finished:
      lower_bound_alive_scores = tf.where(
          beam_state.finished,
          -INF + tf.zeros_like(beam_state.log_probs),
          beam_state.log_probs)
      lower_bound_alive_scores, _ = tf.nn.top_k(lower_bound_alive_scores, 1)
    else:
      lower_bound_alive_scores = beam_state.log_probs[:, 0]
      lower_bound_alive_scores = tf.where(
          beam_state.finished[:, 0],
          -INF + tf.zeros_like(lower_bound_alive_scores),
          beam_state.log_probs[:, 0])

    if alpha > 0:
      max_length_penalty = tf.pow(
          ((5. + tf.cast(maximum_iterations, constant_utils.DT_FLOAT())) / 6.), alpha)
      lower_bound_alive_scores = lower_bound_alive_scores / max_length_penalty

    # Now to compute the lowest score of a finished sequence in finished
    # If the sequence isn't finished, we multiply it's score by 0. since
    # scores are all -ve, taking the min will give us the score of the lowest
    # finished item.
    finished_scores = beam_state.log_probs
    if alpha > 0:
      finished_scores /= bs_utils._calc_length_penalty(beam_state.lengths, alpha)
    if beta > 0:
      finished_scores += bs_utils._calc_coverage_penalty(
          beam_state.attention_scores, beta, sequence_length=beam_state.lengths)

    lowest_score_of_fininshed_in_finished = tf.reduce_min(
        finished_scores * finished_mask, axis=1)
    # If none of the sequences have finished, then the min will be 0 and
    # we have to replace it by -ve INF if it is. The score of any seq in alive
    # will be much higher than -ve INF and the termination condition will not
    # be met.
    lowest_score_of_fininshed_in_finished += (
        (1. - tf.cast(tf.reduce_any(beam_state.finished, 1), constant_utils.DT_FLOAT())) * -INF)

    bound_is_met = tf.reduce_all(
        tf.greater(lowest_score_of_fininshed_in_finished,
                  lower_bound_alive_scores))
    return bound_is_met


  def step(self, time_, logits, beam_state, 
            attention_scores, 
            logits_is_log_prob=False):
    """Performs a single step of Beam Search Decoding.

    Args:
      time_: Beam search time step, should start at 0. At time 0 we assume
        that all beams are equal and consider only the first beam for
        continuations.
      logits: Logits at the current time step. A tensor of shape `[B, vocab_size]`
      beam_state: Current state of the beam search. An instance of `BeamState`
      config: An instance of `BeamSearchConfig`

    Returns:
      A new beam state.
    """

    # some common variables
    beam_width = self.config.beam_width
    alpha = self.config.length_penalty_weight
    beta = self.config.coverage_penalty_weight
    vocab_size = tf.shape(logits)[-1]
    eos_id = self.config.eos_token

    num_beams = self.num_beams
    batch_size = self.batch_size

    # Convert logits to normalized log probs
    if logits_is_log_prob:
      candidate_log_probs = logits
    else:
      candidate_log_probs = bs_utils._log_prob_from_logits(logits)

    # some useful tensors
    inf_tensor = -INF * tf.ones([batch_size, beam_width], dtype=constant_utils.DT_FLOAT())
    zero_tensor = tf.zeros([batch_size, beam_width], dtype=constant_utils.DT_FLOAT())
    inf_tensor = -INF + zero_tensor
    
    # total log probs of current hypothesis
    total_log_probs = candidate_log_probs + tf.expand_dims(beam_state.log_probs, axis=-1)
    # suppress finished beams
    mask_tensor = tf.where(beam_state.finished, inf_tensor, zero_tensor)
    total_log_probs = total_log_probs + tf.expand_dims(mask_tensor, -1)

    # select top-k candidates from distribution
    # reshape: [batch*beam, vocab] => [batch, beam*vocab]
    flatten_scores = tf.reshape(total_log_probs, [batch_size, beam_width * vocab_size])
    # shape: [batch, beam*vocab] => [batch, beam]
    topk_log_probs, indices = tf.nn.top_k(flatten_scores, beam_width)
    
    # tok indices
    beam_indices = tf.div(indices, vocab_size)
    word_indices = tf.mod(indices, vocab_size)
    
    # mask finished indices
    beam_indices = tf.where(
        beam_state.finished, 
        self.init_beam_ids, 
        beam_indices)
    word_indices = tf.where(
        beam_state.finished,  
        tf.zeros_like(word_indices)+eos_id,
        word_indices,)

    # sort according to word ids to put finished sequences to the end
    # not work here because UNK is before </s>
    # word_indices = tf.reshape(word_indices, [batch, beam_width])
    tmp_log_probs = tf.where(
        tf.equal(word_indices, eos_id),
        inf_tensor, topk_log_probs)

    # get real alive indices
    # finished indices are put end
    _, swap_gather_fn = bs_utils._choose_top_k(tmp_log_probs, beam_width, self.batch_indices)
    beam_indices = swap_gather_fn(beam_indices)
    word_indices = swap_gather_fn(word_indices)
    topk_log_probs = swap_gather_fn(topk_log_probs)

    # gather alive tensors
    topk_coordinates = tf.stack([self.batch_indices, beam_indices], axis=2)

    attention_scores = tf.gather_nd(attention_scores, topk_coordinates)
    
    prev_finished = tf.gather_nd(beam_state.finished, topk_coordinates)      
    
    log_probs = tf.gather_nd(beam_state.log_probs, topk_coordinates)
    log_probs = tf.where(prev_finished, log_probs, topk_log_probs)
    
    lengths = tf.gather_nd(beam_state.lengths, topk_coordinates)
    lengths = tf.where(prev_finished, lengths, lengths+1)
    
    candidates = tf.gather_nd(beam_state.sequences, topk_coordinates)
    candidates = tf.concat([candidates, tf.expand_dims(word_indices,-1)], 2)

    alignments = tf.concat([beam_state.attention_scores, tf.expand_dims(attention_scores, 2)], 2)
    alignments = tf.gather_nd(alignments, topk_coordinates)

    # new finished tensor
    finished = tf.equal(word_indices, eos_id)
    
    # outputs
    next_state = BeamSearchState(
        sequences=candidates,
        log_probs=log_probs,
        attention_scores=alignments,
        lengths=lengths,
        finished=finished)

    output = bs_utils.BeamSearchStepOutput(
        predicted_ids=word_indices,
        beam_parent_ids=beam_indices)

    return output, next_state