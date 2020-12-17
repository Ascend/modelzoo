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
Base class for sequence decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import namedtuple
import six

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from noahnmt.layers import common_layers as common_utils
from noahnmt.decoders import decoder
from noahnmt.decoders import helper
from noahnmt.utils import rnn_utils
from noahnmt.utils import registry
from noahnmt.utils import constant_utils
from noahnmt.inference.beam_search_utils import _tile_batch


@registry.register_decoder
class AttentionDecoder(decoder.Decoder):
  """An RNN Decoder that uses attention over an input sequence.
  """

  def __init__(self, params, mode, name="attention_decoder"):
    super(AttentionDecoder, self).__init__(params, mode, name=name)
    self._built = False
  

  @staticmethod
  def default_params():
    params = rnn_utils.default_rnn_cell_params()
    params.update({
      "decode_length_factor": 2.0,
      "attention.class": "sum_attention",
      "attention.params": {},
      "pass_state": True,
      "attention_layer": True,
    })
    return params


  def _output_size(self):
    vocab_size = self.features["target_modality"].vocab_size
    emb_size = self.features["target_modality"].num_units
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      return tf.TensorShape([emb_size])
    elif isinstance(vocab_size, int):
      return tf.TensorShape([vocab_size])
    else:
      return tf.reshape(vocab_size, [-1])


  @property
  def output_size(self):
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      return {
          decoder.LOGITS: self._output_size(),
          decoder.ATTENTION_SCORES: tf.reshape(self.max_src_len, [-1])
      }
    else:
      return {
          decoder.PREDICTED_IDS: tf.TensorShape([]),
          decoder.LOGITS: self._output_size(),
          decoder.ATTENTION_SCORES: tf.reshape(self.max_src_len, [-1])
      }


  @property
  def output_dtype(self):
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      return {
          decoder.LOGITS: constant_utils.DT_FLOAT(),
          decoder.ATTENTION_SCORES: constant_utils.DT_FLOAT()
      }
    else:
      return {
          decoder.PREDICTED_IDS: constant_utils.DT_INT(),
          decoder.LOGITS: constant_utils.DT_FLOAT(),
          decoder.ATTENTION_SCORES: constant_utils.DT_FLOAT()
      }


  def initialize(self, name=None):
    finished, first_inputs = self.helper.initialize()

    # Concat empty attention context
    attention_context = tf.zeros(
        shape=[self.batch_size, self.attention_fn.output_size],
        dtype=constant_utils.DT_FLOAT())

    return finished, first_inputs, (self.initial_state, attention_context)


  def step(self, time_, inputs, state, name=None):
    cell_state, att_context = state

    # perform rnn cell
    inputs = tf.concat([inputs, att_context], 1)
    cell_output, cell_state = self.cell(inputs, cell_state)
    
    # Compute attention
    att_scores, att_context = self.attention_fn(
        query=cell_output)
    
    # extra layer for fusing attention and rnn output
    if self.params["attention_layer"]:
      cell_output = self.attention_layer(
          inputs=tf.concat([cell_output, att_context], 1))

    # in TRAIN mode, no predicted_ids is needed
    outputs = {
        decoder.LOGITS: cell_output,
        decoder.ATTENTION_SCORES: att_scores
    }

    # in EVAL and PREDICT mode, we need to compute softmax at each step
    # and get predicted_ids
    if self.mode != tf.estimator.ModeKeys.TRAIN:
      cell_output = self.features["target_modality"].top(cell_output)

      # predicted ids
      sample_ids = self.helper.sample(
          time=time_, outputs=cell_output, state=state)
      outputs[decoder.PREDICTED_IDS] = sample_ids
      outputs[decoder.LOGITS] = cell_output

    # prepare for the next step
    finished, next_inputs, _ = self.next_inputs(
        time_=time_, 
        outputs=outputs[decoder.LOGITS], 
        state=None, 
        sample_ids=sample_ids
    )
    next_state = (cell_state, att_context)

    return (outputs, next_state, next_inputs, finished)


  def next_inputs(self, time_, outputs, state, sample_ids):
    return self.helper.next_inputs(
            time=time_, outputs=outputs,
            state=state, sample_ids=sample_ids)


  @property
  def batch_size(self):
    return self._batch_size


  def finalize(self, outputs, final_state):
    """Applies final transformation to the decoder output once decoding is
    finished.
    """
    # in train mode, when loop finishs, we calc real logits
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      logits = self.features["target_modality"].top(outputs[decoder.LOGITS])
      outputs[decoder.LOGITS] = logits

    return outputs

  
  def _build_cell(self):
    self.cell = rnn_utils.create_multi_rnn_cell(
        mode=self.mode,
        unit_type=self.params["rnn.cell_type"],
        num_units=self.params["num_units"],
        num_layers=self.params["num_layers"],
        forget_bias=self.params["forget_bias"],
        dropout_rate=self.params["dropout_rate"],
        layer_norm=self.params["layer_norm"],
        residual=self.params["residual"],
        residual_start=self.params["residual.start_layer"],
        residual_fn=self.params["residual.custom_fn"]
    )
  

  def _setup(self, 
             features, 
             encoder_outputs, 
             beam_width=1, 
             use_sampling=False, 
             **kwargs):
    """ setups before dynamic decoding
    """
    self._built = True
    self.features = features
    self.encoder_outputs = encoder_outputs
    self._batch_size = tf.size(encoder_outputs["memory_len"])
    self.max_src_len = tf.shape(features["source_ids"])[1]
    num_units = self.params["num_units"]

    if self.mode != tf.estimator.ModeKeys.PREDICT:
      self.helper = helper.TrainingHelper(
          inputs=features["target_embeded"],
          time_major=False,
          sequence_length=features["target_len"]
      )
      # training help, max_time
      self.maximum_iterations = self.helper.max_time
    elif use_sampling:
      self.helper = helper.SamplingEmbeddingHelper(
          embedding=features["target_modality"].target_bottom_weight * num_units**0.5,
          start_tokens=tf.fill(
              [self.batch_size], 
              features["target_modality"].sos),
          end_token=features["target_modality"].eos
      )
    else:
      self.helper = helper.GreedyEmbeddingHelper(
          embedding=features["target_modality"].target_bottom_weight * num_units**0.5,
          start_tokens=tf.fill(
              [self.batch_size], 
              features["target_modality"].sos),
          end_token=features["target_modality"].eos
      )

    if self.mode == tf.estimator.ModeKeys.PREDICT:
      # max iterations when using dynamic_decode
      self.maximum_iterations = tf.cast(
          tf.round(float(self.params["decode_length_factor"]) * tf.to_float(self.max_src_len)),
          constant_utils.DT_INT())
      
    
    # vocab_size, could be the selected_size
    self.vocab_size = features["target_modality"].vocab_size
    # parallel_iterations, set to 1 when decoding because of the auto-regressive nature
    self.parallel_iterations = 32
    if self.mode == tf.estimator.ModeKeys.PREDICT:
      self.parallel_iterations = 1
    
    # build rnn cell
    self._build_cell()

    # build attention layer
    attention_cls = registry.attention(self.params["attention.class"])
    self.attention_fn = attention_cls(
        params=self.params["attention.params"], 
        mode=self.mode)
    self.attention_fn.prepare_memory(
        memory=encoder_outputs["encoder_output"],
        memory_length=encoder_outputs["memory_len"])
    
    # initial_state
    if self.params["pass_state"]:
      # init state from encoder final state
      self.initial_state = encoder_outputs["final_state"]
      # Tile initial state
      # if beam_width > 1:
      #   self.initial_state = nest.map_structure(
      #     lambda x: _tile_batch(x, beam_width), self.initial_state)
    else:
      # zero initial state
      self.initial_state = self.cell.zero_state(
          batch_size=self.batch_size,
          dtype=constant_utils.DT_FLOAT())
    
    # attention layer
    if self.params["attention_layer"]:
      self.attention_layer = tf.layers.Dense(
          units=self.params["num_units"],
          use_bias=False,
          name="attention_layer")


  def decode(self, features, encoder_outputs, **kwargs):
    """ decode
    """
    # first some setups
    if not self._built:
      self._setup(features, encoder_outputs, **kwargs)

    # dynamic decode
    outputs, final_state = decoder.dynamic_decode(
        decoder=self,
        output_time_major=False,
        impute_finished=False,
        swap_memory=(self.mode == tf.estimator.ModeKeys.TRAIN),
        parallel_iterations=self.parallel_iterations,
        maximum_iterations=self.maximum_iterations)

    # finalize to get final outputs
    return self.finalize(outputs, final_state)
