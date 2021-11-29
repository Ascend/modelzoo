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

"""A decoder that uses beam search. Can only be used for inference, not
training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import tensorflow as tf
from tensorflow.python.util import nest

from noahnmt.inference.beam_search import BeamSearch
from noahnmt.decoders import decoder
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.layers import common_layers
from noahnmt.layers.common_layers import transpose_batch_time as _transpose_batch_time
from noahnmt.inference.beam_search_utils import _tile_batch
from noahnmt.inference.beam_search_utils import _merge_beam_dim
from noahnmt.inference.beam_search_utils import _unmerge_beam_dim
from noahnmt.inference.beam_search_utils import get_shape_invariants

from noahnmt.utils import graph_utils

@registry.register_decoder
class BeamSearchDecoder(decoder.Decoder):
  """The BeamSearchDecoder wraps another decoder to perform beam search instead
  of greedy selection. This decoder must be used with batch size of 1, which
  will result in an effective batch size of `beam_width`.

  Args:
    decoder: A instance of `RNNDecoder` to be used with beam search.
    config: A `BeamSearchConfig` that defines beam search decoding parameters.
  """

  def __init__(self, decoder_, config, search_type=None):
    super(BeamSearchDecoder, self).__init__(
        params={}, 
        mode=tf.estimator.ModeKeys.PREDICT,
        name=decoder_.name)

    self.decoder = decoder_
    self.config = config
    assert self.decoder.mode == tf.estimator.ModeKeys.PREDICT
    
    self._num_beams = None
    self._batch_size = None
    self.beam_search = None
    self._search_type = search_type
    self._built = False

  
  def __call__(self, *args, **kwargs):
    return self._build(*args, **kwargs)

  @property
  def batch_size(self):
    return self._batch_size
  
  @property
  def num_beams(self):
    if self._num_beams is None:
      self._num_beams = self.batch_size * self.beam_width
    return self._num_beams

  @property
  def beam_width(self):
    return self.config.beam_width


  def initialize(self, name=None):
    finished, first_inputs, initial_state = self.decoder.initialize()

    # Create beam state
    batch_size = self.batch_size
    beam_state = self.beam_search.create_initial_beam_state(
                      attention_length=self.attention_length,
                      initial_ids=self.decoder.helper._start_tokens)
    return finished, first_inputs, (initial_state, beam_state)
  

  def finalize(self, final_state):
    sequences, attention_scores = self.beam_search.update_final_state(final_state)
   
    if self.config.return_top_beam:
      sequences = sequences[:,0]
      attention_scores = attention_scores[:,0]
    else:
      # to batch x time x beam
      sequences = tf.transpose(sequences, [0, 2, 1])
      # to batch x time x beam x src_len
      attention_scores = tf.transpose(attention_scores, [0, 2, 1, 3])

    final_outputs={
        decoder.PREDICTED_IDS: sequences,
        decoder.ATTENTION_SCORES: attention_scores
    }

    return final_outputs


  def step(self, time_, inputs, state, name=None):
    decoder_state, beam_state = state

    # Call the original decoder
    (decoder_output, decoder_state, _, _) = self.decoder.step(time_, inputs,
                                                              decoder_state)
    
    attention_scores = decoder_output[decoder.ATTENTION_SCORES]
    attention_scores = _unmerge_beam_dim(
        attention_scores, self.batch_size, self.beam_width)

    cell_output = decoder_output[decoder.LOGITS]
    cell_output = nest.map_structure(
      lambda out: _unmerge_beam_dim(out, self.batch_size, self.beam_width), 
      cell_output)

    # Perform a step of beam search
    bs_output, beam_state = self.beam_search.step(
        time_=time_,
        logits=cell_output,
        beam_state=beam_state,
        attention_scores=attention_scores,
        logits_is_log_prob=False)

    # Shuffle everything according to beam search result
    # bpid_shape = tf.shape(bs_output.beam_parent_ids)
    beam_parent_ids = bs_output.beam_parent_ids + \
                     tf.expand_dims(tf.range(self.batch_size) * self.beam_width, 1)
    beam_parent_ids = tf.reshape(beam_parent_ids, [-1])
    decoder_state = nest.map_structure(
        lambda x: tf.gather(x, beam_parent_ids), decoder_state)
    decoder_output = nest.map_structure(
        lambda x: tf.gather(x, beam_parent_ids), decoder_output)

    next_state = (decoder_state, beam_state)
    finished, next_inputs, next_state = self.decoder.next_inputs(
        time_=time_,
        outputs=decoder_output,
        state=next_state,
        sample_ids=tf.reshape(bs_output.predicted_ids, [-1]))
    # next_inputs.set_shape([self.batch_size, None])

    return (next_state, next_inputs, finished)


  def _setup(self, features, encoder_outputs, **kwargs):
    self._built = True
    self.features = features
    self.encoder_outputs = encoder_outputs

    if isinstance(features, (tuple, list)):
      shape_list = common_layers.shape_list(features[0]["source_ids"])
      self._batch_size = shape_list[0]
      self.attention_length = shape_list[1]
    else:
      shape_list = common_layers.shape_list(features["source_ids"])
      self._batch_size = shape_list[0]
      self.attention_length = shape_list[1]

    if self.config.beam_width > 1:
      if isinstance(encoder_outputs, (tuple, list)):
        for i in range(len(encoder_outputs)):
          for key, value in encoder_outputs[i].items():
            encoder_outputs[i][key] = nest.map_structure(
                lambda t: _tile_batch(t, self.config.beam_width), value)
      else:
        for key, value in encoder_outputs.items():
          graph_utils.add_dict_to_collection({key+"_before_tile": value}, "SAVE_TENSOR")
          encoder_outputs[key] = nest.map_structure(
              lambda t: _tile_batch(t, self.config.beam_width), value)
          # encoder_outputs[key] = _tile_batch(value, self.config.beam_width)
          graph_utils.add_dict_to_collection({key+"_after_tile": encoder_outputs[key]}, "SAVE_TENSOR")
    
    # BeamSearch for beam search step
    self.beam_search = BeamSearch(
        batch_size=self.batch_size,
        config=self.config,
        num_beams=self._num_beams)

    # setup inner decoder
    self.decoder._setup(features, encoder_outputs, beam_width=self.config.beam_width, **kwargs)
    self.maximum_iterations = self.decoder.maximum_iterations


  def decode(self, features, encoder_outputs, **kwargs):
    if not self._built:
      with self.variable_scope():
        self._setup(features, encoder_outputs, **kwargs)

    # final_state = dynamic_decode(
    #     decoder=self,
    #     output_time_major=False,
    #     swap_memory=False,
    #     maximum_iterations=self.maximum_iterations,
    #     parallel_iterations=1,
    #     stop_early=self.config.stop_early)
    final_state = static_decode(
        decoder=self,
        output_time_major=False,
        swap_memory=False,
        maximum_iterations=self.maximum_iterations,
        parallel_iterations=1,
        stop_early=self.config.stop_early)
    return self.finalize(final_state)


def static_decode(decoder,
                  output_time_major=False,
                  maximum_iterations=None,
                  parallel_iterations=1,
                  swap_memory=False,
                  scope=None,
                  reuse=False,
                  stop_early=True):
  """Perform static decoding with `decoder`
  """
  if not isinstance(decoder, BeamSearchDecoder):
    raise TypeError("Expected decoder to be type BeamSearchDecoder, but saw: %s" %
                    type(decoder))

  # with tf.variable_scope(decoder.variable_scope(), reuse=tf.AUTO_REUSE) as varscope:
  with decoder.variable_scope() as varscope:
    # if reuse:
      # varscope.reuse_variables()
    # Properly cache variable values inside the while_loop
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    assert maximum_iterations is not None

    initial_finished, initial_inputs, initial_state = decoder.initialize()

    # if maximum_iterations is not None:
    #   initial_finished = tf.logical_or(
    #       initial_finished, 0 >= maximum_iterations)
    # initial_time = tf.constant(0, dtype=constant_utils.DT_INT())

    

    finished = initial_finished
    inputs = initial_inputs
    state = initial_state

    for time in range(maximum_iterations):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: 1-D bool tensor.

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
      """
      # tf.logging.info("############################LOOP %d#######################"%time)
      # if reuse:
      # varscope.reuse_variables()
      with tf.variable_scope(tf.get_variable_scope(), reuse=time>0):
        (decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs, state)
      next_finished = tf.logical_or(decoder_finished, finished)
      
      # assert decoder.mode == tf.estimator.ModeKeys.PREDICT
      next_finished = tf.logical_or(
          next_finished, time + 1 >= maximum_iterations)

      finished = next_finished
      inputs = next_inputs
      state = decoder_state

    
    # final_outputs_ta = res[1]
    final_decoder_state, final_beam_state = state
    # if decoder.config.coverage_penalty_weight > 0:
    return final_beam_state



def dynamic_decode(decoder,
                  output_time_major=False,
                  maximum_iterations=None,
                  parallel_iterations=1,
                  swap_memory=False,
                  scope=None,
                  reuse=False,
                  stop_early=True):
  """Perform dynamic decoding with `decoder`.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.

  Returns:
    `(final_outputs, final_state)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if maximum_iterations is provided but is not a scalar.
  """
  if not isinstance(decoder, BeamSearchDecoder):
    raise TypeError("Expected decoder to be type BeamSearchDecoder, but saw: %s" %
                    type(decoder))

  # with tf.variable_scope(scope or "decoder", reuse=reuse) as varscope:
  with decoder.decoder.variable_scope() as varscope:
    if reuse:
      varscope.reuse_variables()
    # Properly cache variable values inside the while_loop
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    assert maximum_iterations is not None

    initial_finished, initial_inputs, initial_state = decoder.initialize()

    if maximum_iterations is not None:
      initial_finished = tf.logical_or(
          initial_finished, 0 >= maximum_iterations)
    initial_time = tf.constant(0, dtype=constant_utils.DT_INT())


    def condition(unused_time, state, unused_inputs,
                  finished):
      if not stop_early:
        return tf.logical_not(tf.reduce_all(finished))
      
      decoder_state, beam_state = state
      bound_is_met  = decoder.beam_search.bound_is_met(beam_state, maximum_iterations)

      return tf.logical_and(
          tf.logical_not(tf.reduce_all(finished)), tf.logical_not(bound_is_met))

    def body(time, state, inputs, finished):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: 1-D bool tensor.

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
      """
      (decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs, state)
      next_finished = tf.logical_or(decoder_finished, finished)
      
      # assert decoder.mode == tf.estimator.ModeKeys.PREDICT
      next_finished = tf.logical_or(
          next_finished, time + 1 >= maximum_iterations)

      nest.assert_same_structure(inputs, next_inputs)

      return (time + 1, decoder_state, next_inputs, next_finished)
    
    loop_vars = [
        initial_time, initial_state, initial_inputs,
        initial_finished]
    shape_invariants = get_shape_invariants(loop_vars)

    res = tf.while_loop(
        condition,
        body,
        loop_vars=loop_vars,
        shape_invariants=shape_invariants,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory, back_prop=False)

    # final_outputs_ta = res[1]
    final_decoder_state, final_beam_state = res[1]
    # if decoder.config.coverage_penalty_weight > 0:
    return final_beam_state
