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
Base class for transformer decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six
from pydoc import locate

import tensorflow as tf
from tensorflow.python.util import nest

from noahnmt.decoders import helper
from noahnmt.decoders import decoder
from noahnmt.decoders.decoder import get_state_shape_invariants
from noahnmt.decoders.attention_decoder import AttentionDecoder
from noahnmt.layers.common_layers import transpose_batch_time as _transpose_batch_time
from noahnmt.utils import transformer_utils as t2t_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.utils import optimize_utils
from noahnmt.layers import common_layers
from noahnmt.utils import graph_utils

@registry.register_decoder
class TransformerDecoder(AttentionDecoder):
  """An self-attention Decoder https://arxiv.org/abs/1706.03762.
  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self, params, mode, name="transformer_decoder"):
    super(TransformerDecoder, self).__init__(params, mode, name)
    self._built = False
    self._combiner_fn = locate(self.params["position.combiner_fn"])

    
  @staticmethod
  def default_params():
    return {
        "num_layers": 6,
        "num_units": 512,
        "layer.preprocess": "n",
        "layer.postprocess": "da",
        "attention.self_average": False,
        "attention.num_heads": 8,
        "attention.branch": False,
        "attention.relpos": 0,
        "ffn.num_units": 2048,
        "ffn.activation": "relu",
        "dropout_rate": 0.1,
        "position.enable": True,
        "position.combiner_fn": "tensorflow.add",
        "position.max_length": 500,
        "decode_length_factor": 2.,
        "flex_decode_length": True,
        "initializer": "uniform_unit_scaling",
        "init_scale": 1.0,
        "attention.weighted_avg": False
    }
  

  def initialize(self, name=None):
    tf.get_variable_scope().set_initializer(
          optimize_utils.get_variable_initializer(
              initializer=self.params["initializer"],
              initializer_gain=self.params["init_scale"]))
    print("*" * 100)
    finished, first_inputs = self.helper.initialize()
    return finished, first_inputs, self.initial_state
  

  def finalize(self, outputs, final_state):
    return outputs
  

  def get_shape_invariants(self, loop_vars):
    return nest.map_structure(get_state_shape_invariants, loop_vars)
  

  def _add_position_embedding(self, inputs, time):
    x_type = inputs.dtype
    inputs = tf.cast(inputs, tf.float32)
    pos_embed = self.pos_embed[time, :]
    # now shape [x, time, units]
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    # seq_pos_embed_batch = tf.tile(seq_pos_embed, [self.config.beam_width,1,1])
    # assume inside combiner_fn, pos_embed will be automatically broadcasted along the batch-dim
    output = self._combiner_fn(inputs, pos_embed)
    output = tf.cast(output, x_type)
    return output


  def step(self, time_, inputs, state, name=None):
    # currently, step is used only in PREDICT mode
    # so no dropout inside this fucntion
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    # in every step, inputs is in full length
    # from time+1, inputs are all zero-paddings
    # cur_inputs = tf.expand_dims(inputs, 1)
    cur_inputs = inputs * (self.params["num_units"] ** 0.5)

    # zeros_padding = inputs[:, time_ + 2:, :] 
    cur_inputs_pos = self._add_position_embedding(cur_inputs, time_)
    cur_inputs_pos = tf.cast(cur_inputs_pos, constant_utils.DT_FLOAT())
    # transformer
    cell_outputs, att_scores, att_context, state = self._transformer_block(
        inputs=cur_inputs_pos,
        encoder_mask=self.encoder_mask,
        decoder_mask=None,
        decoder_memory=None,
        cache=state,
        position=time_)
    
    # squeeze
    # output is in shape: [batch, 1, units]
    # cell_outputs = tf.squeeze(cell_outputs, axis=1)
    att_scores = tf.squeeze(att_scores, axis=1)
    # att_context = tf.squeeze(att_context, axis=1)
    # state = tf.squeeze(state, axis=1)
     
    sample_ids = self.helper.sample(
        time=time_, outputs=cell_outputs, state=state)

    outputs = {
        decoder.LOGITS: cell_outputs,
        decoder.PREDICTED_IDS: sample_ids,
        decoder.ATTENTION_SCORES: att_scores
    }

    finished, next_inputs, _ = self.next_inputs(
        time_=time_, outputs=outputs[decoder.LOGITS], 
        state=state, sample_ids=sample_ids)

    return outputs, state, next_inputs, finished
  
  
  def _attention_ffn_block(self, inputs, encoder_mask, decoder_mask, decoder_memory=None, cache=None, position=None):
    is_training = self.mode == tf.estimator.ModeKeys.TRAIN
    is_infer = self.mode == tf.estimator.ModeKeys.PREDICT
    # with tf.variable_scope("att_ffn_block"):
    x = inputs
    attention = None
    for layer_idx in range(self.params["num_layers"]):
      with tf.variable_scope("layer_%d" % layer_idx):
        ## the first sublayer is self-attention
        # first apply layer_norm
        # then multi-head attention
        # then dropout and residual
        key_name = "layer_%d" % layer_idx
        layer_cache = cache[key_name] if cache is not None else None
        decoder_memory = None

        if self.params["attention.self_average"]:
          # average attention network as in https://arxiv.org/pdf/1805.00631.pdf
          with tf.variable_scope("average_attention"):
            y = t2t_utils.average_attention_sublayer(
                  x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]),
                  mask=decoder_mask,
                  num_units=self.params["num_units"],
                  dropout_rate=self.params["dropout_rate"],
                  is_training=is_training,
                  cache=layer_cache,
                  filter_depth=self.params["ffn.num_units"],
                  activation=self.params["ffn.activation"],
                  position=position,
              )
            x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
        else:
          with tf.variable_scope("self_attention"):
            _, y = t2t_utils.self_attention_sublayer(
                x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]),
                mask=decoder_mask,
                num_units=self.params["num_units"],
                num_heads=self.params["attention.num_heads"],
                dropout_rate=self.params["dropout_rate"],
                is_training=is_training,
                memory=decoder_memory,
                cache=layer_cache,
                branch=self.params["attention.branch"],
                filter_depth=self.params["ffn.num_units"],
                activation=self.params["ffn.activation"],
                relpos=self.params["attention.relpos"],
                batch_size=self._batch_size,
                from_seq_len=self.from_seq_len,
                to_seq_len=self.from_seq_len,
            )
            # x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
            x = t2t_utils.layer_process(y, y=x, dropout_rate=self.params["dropout_rate"], is_training=is_training, mode=self.params["layer.postprocess"])

        ## the second sublayer is attention to encoder
        with tf.variable_scope("encdec_attention"):
          att_scores, y = t2t_utils.self_attention_sublayer(
              x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]),
              mask=encoder_mask,
              num_units=self.params["num_units"],
              num_heads=self.params["attention.num_heads"],
              dropout_rate=self.params["dropout_rate"],
              is_training=is_training,
              memory=self.encoder_outputs["encoder_output"],
              branch=self.params["attention.branch"],
              filter_depth=self.params["ffn.num_units"],
              activation=self.params["ffn.activation"],
              cache=layer_cache,
              batch_size=self._batch_size,
              from_seq_len=self.from_seq_len,
              to_seq_len=self.max_src_len,
          )
          # x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
          x = t2t_utils.layer_process(y, y=x, dropout_rate=self.params["dropout_rate"], is_training=is_training, mode=self.params["layer.postprocess"])

          att_context = x

          if not self.params["attention.weighted_avg"]:
            att_scores = tf.reduce_mean(att_scores, axis=1)

          if attention is None:
            attention = att_scores
          else:
            if not self.params["attention.weighted_avg"]:
              attention += att_scores
            else:
              attention = tf.concat([attention, att_scores], axis=1)

        # the third sublayer is feed-forward
        # first layer_norm
        # then feed-forward
        # then dropout and residual
        with tf.variable_scope("ffn"):
          y = t2t_utils.feed_forward_sublayer(
              x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]),
              filter_depth=self.params["ffn.num_units"],
              num_units=self.params["num_units"],
              dropout_rate=self.params["dropout_rate"],
              is_training=is_training,
              activation=self.params["ffn.activation"]
          )
          # x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
          x = t2t_utils.layer_process(y, y=x, dropout_rate=self.params["dropout_rate"], is_training=is_training, mode=self.params["layer.postprocess"])
        
    # shape: [batch, time, units]
    # with tf.variable_scope("outputs"):
    #   outputs = t2t_utils.layer_norm(x)
    outputs = t2t_utils.layer_process(x, mode=self.params["layer.preprocess"])

    if not self.params["attention.weighted_avg"]:
      attention /= self.params["num_layers"]
    else:
      att_alpha = tf.get_variable(
          name="att_alpha",
          shape=[self.params["num_layers"]*self.params["attention.num_heads"]],
          dtype=constant_utils.DT_FLOAT())
      att_alpha = tf.reshape(tf.nn.softmax(att_alpha), shape=[1,-1,1,1])
      attention = tf.reduce_sum(attention * att_alpha, axis=1)

    return outputs, attention, att_context


  def _transformer_block(self, inputs, encoder_mask, decoder_mask, decoder_memory=None, cache=None, position=None):
    # with tf.variable_scope("transformer_block"):    
      # next_layer = inputs
    outputs, att_scores, att_context = self._attention_ffn_block(
        inputs=inputs,
        encoder_mask=encoder_mask,
        decoder_mask=decoder_mask,
        decoder_memory=decoder_memory,
        cache=cache,
        position=position)

    # logits
    outputs = self.features["target_modality"].top(outputs)
      
    return outputs, att_scores, att_context, cache  
  

  def _build_train(self, features, encoder_outputs, **kwargs):
    assert self.mode != tf.estimator.ModeKeys.PREDICT
    
    if not self._built:
      self._setup(features, encoder_outputs, **kwargs)

    # inputs from T2TTrainingHelper is batch-major
    inputs = self.features["target_embeded"]

    graph_utils.add_dict_to_collection({"target_input": inputs}, "SAVE_TENSOR")
	
    input_shape = common_layers.shape_list(inputs)

    if self.params["position.enable"]:
      # x_type = inputs.dtype
      # inputs = tf.cast(inputs, tf.float32)
      positions_embed = t2t_utils.position_encoding(
          length=128, #input_shape[1],
          depth=input_shape[-1] // 2)
      # positions_embed = self._pos_embed[:tf.shape(inputs)[1],:]
      position_ids = features.get("target_posids", None)
      if position_ids is not None:
        positions_embed = common_layers.gather_npu(positions_embed, position_ids)
        inputs = inputs + positions_embed
      else:
        positions_embed = positions_embed[:input_shape[1]]
        inputs = self._combiner_fn(inputs, tf.expand_dims(positions_embed,0))
      inputs = tf.cast(inputs, constant_utils.DT_FLOAT())
      # inputs = self._helper.combine_pos_embed(self._combiner_fn)
    
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])
     
    # Apply dropout to embeddings
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # inputs = tf.layers.dropout(
      #     inputs=inputs,
      #     rate=self.params["dropout_rate"],
      #     training=True)
      inputs = common_layers.npu_dropout(
          input_tensor=inputs,
          dropout_prob=self.params["dropout_rate"])
    # apply layer_norm
    # inputs = t2t_utils.layer_norm(inputs)

    # create masks for encoder outputs and decoder inputs
    # decoder_mask = tf.sequence_mask(
    #     lengths=tf.to_int32(self.features["target_len"]),
    #     maxlen=tf.shape(inputs)[1],
    #     dtype=tf.float32)
    decoder_mask = tf.cast(self.features["target_mask"], constant_utils.DT_FLOAT())
    # [batch, length, length]
    decoder_mask = tf.expand_dims(decoder_mask, axis=-1)
    
    encdec_mask = tf.expand_dims(tf.cast(self.encoder_mask, constant_utils.DT_FLOAT()), -1)
    encdec_mask = tf.matmul(decoder_mask, encdec_mask, transpose_b=True)

    source_segids = features.get("source_segids", None)
    target_segids = features.get("target_segids", None)

    if source_segids is not None:
      source_segids = tf.cast(source_segids, constant_utils.DT_FLOAT())
      target_segids = tf.cast(target_segids, constant_utils.DT_FLOAT())
      cross_mask = tf.expand_dims(target_segids, -1) - tf.expand_dims(source_segids, 1)
      cross_mask = tf.clip_by_value(tf.abs(cross_mask), 0, 1)
      cross_mask = 1 - cross_mask
      encdec_mask = encdec_mask * cross_mask


    decoder_mask = tf.matmul(
        decoder_mask, decoder_mask,
        transpose_b=True)
    # in addition to length mask, we also need to mask future inputs
    # therefore we add an triangle mask
    # 1 0 0
    # 1 1 0
    # 1 1 1
    decoder_mask = tf.matrix_band_part(
        input=decoder_mask,
        num_lower=-1,
        num_upper=0)
    
    
    if target_segids is not None:
      # create mask to avoid cross-segment attention
      target_segids = tf.cast(target_segids, constant_utils.DT_FLOAT())
      self_mask = tf.expand_dims(target_segids, -1) - tf.expand_dims(target_segids, 1)
      self_mask = tf.clip_by_value(tf.abs(self_mask), 0, 1)
      self_mask = 1 - self_mask
      decoder_mask = decoder_mask * self_mask
    
    # decoder_mask = tf.cast(decoder_mask, constant_utils.DT_FLOAT())

    # for s in common_layers.shape_list(inputs):
    #   assert isinstance(s, int)

    # decode using transformer nets
    # return cell_output, attention_scores, attention_context
    # all outputs are batch-major
    cell_output, att_scores, att_context, _ = self._transformer_block(
        inputs=inputs,
        encoder_mask=encdec_mask,
        decoder_mask=decoder_mask)
    # # transpose to time-major [time, batch, units]
    # cell_output = _transpose_batch_time(cell_output)
    # # transpose attention to time-major
    # att_scores = _transpose_batch_time(att_scores)
    # att_context = _transpose_batch_time(att_context)

    # cell_output = tf.reshape(cell_output, input_shape[:-1]+[-1])

    output = {
        decoder.LOGITS: cell_output,
        decoder.ATTENTION_SCORES: att_scores
    }
    if self.mode == tf.estimator.ModeKeys.EVAL:
      output["predicted_ids"] = tf.cast(
          tf.argmax(cell_output, axis=-1), constant_utils.DT_INT())
    return output


  def _build_infer(self, features, encoder_outputs, **kwargs):
    if not self._built:
      with self.variable_scope():
        self._setup(features, encoder_outputs, **kwargs)

    outputs, final_state = decoder.dynamic_decode(
        decoder=self,
        output_time_major=False,
        impute_finished=False,
        swap_memory=True,
        maximum_iterations=self.maximum_iterations)
    
    return self.finalize(outputs, final_state)


  def _setup(self, features, encoder_outputs, beam_width=1, 
             use_sampling=False, **kwargs):
    """
    """            
    self._built = True
    self.features = features
    self.encoder_outputs = encoder_outputs

    enc_shape = common_layers.shape_list(encoder_outputs["memory_mask"])

    self._batch_size = enc_shape[0]
    self.max_src_len = enc_shape[1]
    
    enc_shape = common_layers.shape_list(self.encoder_outputs["encoder_output"])
    # for s in enc_shape:
    #   assert isinstance(s, int)


    self.encoder_outputs["encoder_output"] = tf.reshape(
        self.encoder_outputs["encoder_output"], [enc_shape[0]*enc_shape[1], enc_shape[2]])

    self.from_seq_len = 1
    if "target_ids" in features and self.mode == tf.estimator.ModeKeys.TRAIN:
      self.from_seq_len = common_layers.shape_list(features["target_ids"])[1]
    
    # max_src_len = tf.shape(features["source_ids"])[1]
    # self.max_src_len = max_src_len

    # self.batch_size = input_shape[0]
    # self.from_seq_len = input_shape[1]
    # self.enc_seq_len = common_layers.shape_list(self.encoder_mask)[1]
    
    if self.mode != tf.estimator.ModeKeys.PREDICT:
      self.helper = None
    elif use_sampling:
      self.helper = helper.SamplingEmbeddingHelper(
          embedding=features["target_modality"].target_bottom_weight,
          start_tokens=tf.fill(
              [self.batch_size], 
              features["target_modality"].sos),
          end_token=features["target_modality"].eos
      )
    else:
      self.helper = helper.GreedyEmbeddingHelper(
          embedding=features["target_modality"].target_bottom_weight,
          start_tokens=tf.fill(
              [self.batch_size], 
              features["target_modality"].sos),
          end_token=features["target_modality"].eos
      )

    if self.mode == tf.estimator.ModeKeys.PREDICT:
      # pre-calculated position embedding
      self.pos_embed = t2t_utils.position_encoding(
          length=self.params["position.max_length"],
          depth=self.params["num_units"] // 2)

      # max dynamic_decode steps
      # self.maximum_iterations = tf.cast(
      #     tf.round(float(self.params["decode_length_factor"]) * tf.to_float(max_src_len)),
      #     constant_utils.DT_INT())
      # self.maximum_iterations = tf.minimum(self.maximum_iterations, self.params["position.max_length"])
      self.maximum_iterations = self.params["position.max_length"]

    # encoder mask
    # self.encoder_mask = tf.sequence_mask(
    #     lengths=encoder_outputs["memory_len"],
    #     maxlen=max_src_len,
    #     dtype=constant_utils.DT_FLOAT()
    # )
    self.encoder_mask = tf.cast(encoder_outputs["memory_mask"], constant_utils.DT_FLOAT())

    # cache for fast decoding
    if self.mode == tf.estimator.ModeKeys.PREDICT:
      units = self.params["num_units"]
      if self.params["attention.self_average"]:
        self.initial_state = {
            "layer_%d" % layer: {
                "accum_x": tf.zeros([self.batch_size, 1, units], dtype=constant_utils.DT_FLOAT())
            }
            for layer in range(self.params["num_layers"])
        }
      else:
        units = self.params["num_units"]
        num_heads = int(self.params["attention.num_heads"])
        self.initial_state = {
            "layer_%d" % layer: {
                "k": tf.zeros([self.batch_size, num_heads, 1, units//num_heads], dtype=constant_utils.DT_FLOAT()),
                "v": tf.zeros([self.batch_size, num_heads, 1, units//num_heads], dtype=constant_utils.DT_FLOAT()),
            }
            for layer in range(self.params["num_layers"])
        }

        # precompute endec kv
        for layer in range(self.params["num_layers"]):
          with tf.variable_scope("layer_%d/encdec_attention/multihead_attention" % layer):
            k, v = t2t_utils.precompute_encdec_kv(
                memory=self.encoder_outputs["encoder_output"],
                key_depth=units,
                value_depth=units,
                num_heads=num_heads,
                batch_size=self._batch_size,
                seq_len=self.max_src_len)
          self.initial_state["layer_%d" % layer].update({
              "k_encdec": k,
              "v_encdec": v,
          })
  

  def __call__(self, *args, **kwargs):
    return self._build(*args, **kwargs)


  def decode(self, features, encoder_outputs, **kwargs):
    # scope = tf.get_variable_scope()
    # scope.set_initializer(tf.random_uniform_initializer(
    #     -self.params["init_scale"],
    #     self.params["init_scale"]))
    # if not self._built:
    #   self._setup(features, encoder_outputs)

    # we need to build separate graphs for predict and train
    # because during training we have whole sentences to do parallal conv
    # this is possible because conv_seq2seq doesn't do attention_feed
    # therefor during training there's no dependencies between target inputs
    if self.mode == tf.estimator.ModeKeys.PREDICT:
      return self._build_infer(features, encoder_outputs, **kwargs)
    else:
      # when infer, dynamic decode will add decoder scope, 
      # so we add here to keep it the same
      with self.variable_scope():
        return self._build_train(features, encoder_outputs, **kwargs)
