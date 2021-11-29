## coding=utf-8
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
An encoder that pools over embeddings, as described in
https://arxiv.org/abs/1611.02344.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydoc import locate
import numpy as np
import pdb
import tensorflow as tf

from noahnmt.encoders import encoder
from noahnmt.utils import transformer_utils as t2t_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.utils import optimize_utils
from noahnmt.layers import common_layers as common_utils
from noahnmt.utils import graph_utils

@registry.register_encoder
class TransformerEncoder(encoder.Encoder):
  """A transformer encoder, as described in
  https://arxiv.org/abs/1706.03762. The encoder supports optional positions
  embeddings.
  Params:
    attention_cnn.units: Number of units in `cnn_a`. Same in each layer.
    attention_cnn.kernel_size: Kernel size for `cnn_a`.
    attention_cnn.layers: Number of layers in `cnn_a`.
    embedding_dropout_keep_prob: Dropout keep probability
      applied to the embeddings.
    output_cnn.units: Number of units in `cnn_c`. Same in each layer.
    output_cnn.kernel_size: Kernel size for `cnn_c`.
    output_cnn.layers: Number of layers in `cnn_c`.
    position_embeddings.enable: If true, add position embeddings to the
      inputs before pooling.
    position_embeddings.combiner_fn: Function used to combine the
      position embeddings with the inputs. For example, `tensorflow.add`.
    position_embeddings.num_positions: Size of the position embedding matrix.
      This should be set to the maximum sequence length of the inputs.
  """

  def __init__(self, params, mode, name="transformer_encoder"):
    super(TransformerEncoder, self).__init__(params, mode, name)
    self._combiner_fn = locate(self.params["position.combiner_fn"])


  @staticmethod
  def default_params():
    return {
        "num_units": 512,
        "num_layers": 6,
        "layer.preprocess": "n",
        "layer.postprocess": "da",
        "ffn.num_units": 2048,
        "ffn.activation": "relu", # relu or swish
        "attention.num_heads": 8,
        "attention.branch": False, # weighted transformer in https://arxiv.org/pdf/1711.02132.pdf
        "attention.relpos": 0, # relative position representation in https://arxiv.org/pdf/1803.02155.pdf
        "dropout_rate": 0.1,
        "position.enable": True,
        "position.combiner_fn": "tensorflow.add",
        "initializer":  "uniform_unit_scaling",
        "init_scale": 1.0,
        "share_level": 1 # every 2 layers share the same params
    }

  def encode(self, inputs, mask, segment_ids, position_ids):
    # scope = tf.get_variable_scope()
    # # TODO check initializer in tensor2tensor
    # scope.set_initializer(tf.random_uniform_initializer(
    #     -self.params["init_scale"],
    #     self.params["init_scale"]))

    tf.get_variable_scope().set_initializer(
        optimize_utils.get_variable_initializer(
            initializer=self.params["initializer"],
            initializer_gain=self.params["init_scale"]))
    print("*"*100)
    tf.logging.info("Finish initializing encoder")

    graph_utils.add_dict_to_collection({"encoder_sequence_mask": mask}, "SAVE_TENSOR")

    input_shape = common_utils.shape_list(inputs)
    batch_size = input_shape[0]
    max_len = input_shape[1]

    # for s in input_shape:
    #   assert isinstance(s, int)

    # sequence_mask shape: [batch, length]
    # sequence_mask = tf.sequence_mask(
    #       lengths=tf.to_int32(sequence_length),
    #       maxlen=max_len,
    #       dtype=constant_utils.DT_FLOAT())
    sequence_mask = tf.expand_dims(tf.cast(mask, constant_utils.DT_FLOAT()),-1)
    sequence_mask = tf.matmul(sequence_mask, sequence_mask, transpose_b=True)

    graph_utils.add_dict_to_collection({"sequence_mask": sequence_mask}, "SAVE_TENSOR")

    if segment_ids is not None:
      # create mask to avoid cross-segment attention
      segment_ids = tf.cast(segment_ids, constant_utils.DT_FLOAT())

      self_mask = tf.expand_dims(segment_ids, -1) - tf.expand_dims(segment_ids, 1)

      graph_utils.add_dict_to_collection({"self_mask_init": self_mask}, "SAVE_TENSOR")

      self_mask = tf.clip_by_value(tf.abs(self_mask), 0, 1)

      graph_utils.add_dict_to_collection({"self_mask_clip": self_mask}, "SAVE_TENSOR")

      # self_mask = tf.cast(self_mask, constant_utils.DT_FLOAT())

      self_mask = 1 - self_mask

      graph_utils.add_dict_to_collection({"self_mask_sub": self_mask}, "SAVE_TENSOR")

      sequence_mask = sequence_mask * self_mask

      graph_utils.add_dict_to_collection({"sequence_mask_new": sequence_mask}, "SAVE_TENSOR")


    # NOTE
    # In tensor2tensor, target language id embedding is added to inputs
    word_embed_size = input_shape[-1] #common_utils.shape_list(inputs)[-1]
    #target_space_emb = tf.get_variable("target_space_emb", [word_embed_size], dtype=constant_utils.DT_FLOAT())
    #inputs += tf.reshape(target_space_emb, [1, 1, word_embed_size])

    #graph_utils.add_dict_to_collection({"encoder_emb_before_posemb": inputs}, "SAVE_TENSOR")

    if self.params["position.enable"]:
      # combine input and position embedding
      # returned pos_enc shape: [length, units]
      # x_dtype = inputs.dtype
      # inputs = tf.cast(inputs, tf.float32)
      
      #graph_utils.add_dict_to_collection({"encoder_input_fp32": inputs}, "SAVE_TENSOR")

      pos_enc = t2t_utils.position_encoding(
          length=128, #max_len,
          depth=word_embed_size // 2)
      if not isinstance(max_len, int):
        pos_enc = pos_enc[:max_len]
      
      #graph_utils.add_dict_to_collection({"encoder_pos_enc": pos_enc}, "SAVE_TENSOR")
      
      if position_ids is not None:
        pos_enc = common_utils.gather_npu(pos_enc, position_ids)
      else:
        pos_enc = tf.expand_dims(pos_enc,0)

      #graph_utils.add_dict_to_collection({"encoder_pos_enc_expand": pos_enc}, "SAVE_TENSOR")

      #pos_enc = tf.tile(pos_enc, [batch_size, 1, 1],name='ccggll')

      #graph_utils.add_dict_to_collection({"encoder_pos_enc_tile": pos_enc}, "SAVE_TENSOR")

      inputs = inputs + pos_enc

      # inputs = self._combiner_fn(inputs, tf.expand_dims(pos_enc,0))

      #graph_utils.add_dict_to_collection({"encoder_after_add_pos": inputs}, "SAVE_TENSOR")

      inputs = tf.cast(inputs, constant_utils.DT_FLOAT())

      #graph_utils.add_dict_to_collection({"encoder_after_pos_cast": inputs}, "SAVE_TENSOR")
    
    # inputs  = inputs * tf.expand_dims(sequence_mask, -1)
    #graph_utils.add_dict_to_collection({"encoder_emb_processed": inputs}, "SAVE_TENSOR")

    # input_shape = common_utils.shape_list(inputs)
    inputs = tf.reshape(inputs, [-1, word_embed_size])


    is_training = self.mode == tf.estimator.ModeKeys.TRAIN
    dropout_rate = self.params["dropout_rate"]
    # first apply dropout to inputs
    if is_training:
      # inputs = tf.layers.dropout(
      #     inputs=inputs,
      #     rate=dropout_rate,
      #     training=is_training)
      inputs = common_utils.npu_dropout(
          input_tensor=inputs,
          dropout_prob=dropout_rate)


    # map to hidden size if necessary
    if word_embed_size != self.params["num_units"]:
        inputs = tf.layers.dense(
            inputs, self.params["num_units"], use_bias=True, activation=None,
            name="embedding_hidden_mapping_in")
    
    graph_utils.add_dict_to_collection({"encoder_input": inputs}, "SAVE_TENSOR")

    x = inputs
    #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/encoder_emb_processed21": x}, "SAVE_TENSOR")
    #pdb.set_trace()
    for layer_idx in range(self.params["num_layers"]):
      layer_idx = layer_idx // self.params["share_level"]
      with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("self_attention"):
          #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/encoder_emb_processed22": x}, "SAVE_TENSOR")
          #x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"])
          # x = t2t_utils.layer_norm(x)
          # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "/encoder_emb_processed24": x}, "SAVE_TENSOR")
          _, y = t2t_utils.self_attention_sublayer(
              x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]), 
              mask=sequence_mask,
              num_units=self.params["num_units"],
              num_heads=self.params["attention.num_heads"],
              dropout_rate=dropout_rate,
              is_training=is_training,
              branch=self.params["attention.branch"],
              filter_depth=self.params["ffn.num_units"],
              activation=self.params["ffn.activation"],
              relpos=self.params["attention.relpos"],
              batch_size=batch_size,
              from_seq_len=max_len,
              to_seq_len=max_len,
          )
          #graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_encoder_emb_processed24": x}, "SAVE_TENSOR")
          # graph_utils.add_dict_to_collection({tf.get_variable_scope().name + "_encoder_emb_processed25": y}, "SAVE_TENSOR")
          # x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
          x = t2t_utils.layer_process(y, y=x, dropout_rate=dropout_rate, is_training=is_training, mode=self.params["layer.postprocess"])
          graph_utils.add_dict_to_collection({tf.get_variable_scope().name: x}, "SAVE_TENSOR")

        # the second sublayer is feed-forward
        # first layer_norm
        # then feed-forward
        # then dropout and residual
        with tf.variable_scope("ffn"):
          y = t2t_utils.feed_forward_sublayer(
              x=t2t_utils.layer_process(x, mode=self.params["layer.preprocess"]),
              filter_depth=self.params["ffn.num_units"],
              num_units=self.params["num_units"],
              dropout_rate=dropout_rate,
              is_training=is_training,
              activation=self.params["ffn.activation"]
          )
          
          # x = t2t_utils.layer_process(y, y=x, mode=self.params["layer.postprocess"])
          x = t2t_utils.layer_process(y, y=x, dropout_rate=dropout_rate, is_training=is_training, mode=self.params["layer.postprocess"])
          graph_utils.add_dict_to_collection({tf.get_variable_scope().name: x}, "SAVE_TENSOR")
    
    # shape: [batch, time, units]
    outputs = t2t_utils.layer_process(x, mode=self.params["layer.preprocess"])
    # final_state = tf.reduce_mean(outputs, axis=1)

    outputs = tf.reshape(outputs, [batch_size, max_len, word_embed_size])

    # for s in common_utils.shape_list(outputs):
    #   assert isinstance(s, int)

    graph_utils.add_dict_to_collection({"encoder_output": outputs}, "SAVE_TENSOR")

    return {
        encoder.ENCODER_OUTPUT: outputs,
    }
