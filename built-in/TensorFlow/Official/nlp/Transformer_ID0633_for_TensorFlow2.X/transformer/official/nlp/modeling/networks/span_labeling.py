# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Span labeling network."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


def _apply_position_mask(logits, position_mask):
  """Applies a position mask to calculated logits."""
  if tf.rank(logits) != tf.rank(position_mask):
    position_mask = position_mask[:, None, :]
  masked_logits = logits * (1 - position_mask) - 1e30 * position_mask
  return tf.nn.log_softmax(masked_logits, -1), masked_logits


@tf.keras.utils.register_keras_serializable(package='Text')
class SpanLabeling(tf.keras.Model):
  """Span labeling network head for BERT modeling.

  This network implements a simple single-span labeler based on a dense layer.
  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Arguments:
    input_width: The innermost dimension of the input tensor to this network.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The initializer for the dense layer in this network. Defaults
      to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               input_width,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):
    self._self_setattr_tracking = False
    self._config = {
        'input_width': input_width,
        'activation': activation,
        'initializer': initializer,
        'output': output,
    }

    sequence_data = tf.keras.layers.Input(
        shape=(None, input_width), name='sequence_data', dtype=tf.float32)

    intermediate_logits = tf.keras.layers.Dense(
        2,  # This layer predicts start location and end location.
        activation=activation,
        kernel_initializer=initializer,
        name='predictions/transform/logits')(
            sequence_data)
    self.start_logits, self.end_logits = (
        tf.keras.layers.Lambda(self._split_output_tensor)(intermediate_logits))

    start_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(
        self.start_logits)
    end_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(
        self.end_logits)

    if output == 'logits':
      output_tensors = [self.start_logits, self.end_logits]
    elif output == 'predictions':
      output_tensors = [start_predictions, end_predictions]
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    super(SpanLabeling, self).__init__(
        inputs=[sequence_data], outputs=output_tensors, **kwargs)

  def _split_output_tensor(self, tensor):
    transposed_tensor = tf.transpose(tensor, [2, 0, 1])
    return tf.unstack(transposed_tensor)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


class XLNetSpanLabeling(tf.keras.layers.Layer):
  """Span labeling network head for XLNet on SQuAD2.0.

  This networks implements a span-labeler based on dense layers and question
  possibility classification. This is the complex version seen in the original
  XLNet implementation.

  This applies a dense layer to the input sequence data to predict the start
  positions, and then uses either the true start positions (if training) or
  beam search to predict the end positions.

  **Note: `compute_with_beam_search` will not work with the Functional API
  (https://www.tensorflow.org/guide/keras/functional).

  Arguments:
    input_width: The innermost dimension of the input tensor to this network.
    start_n_top: Beam size for span start.
    end_n_top: Beam size for span end.
    activation: The activation, if any, for the dense layer in this network.
    dropout_rate: The dropout rate used for answer classification.
    initializer: The initializer for the dense layer in this network. Defaults
      to a Glorot uniform initializer.
  """

  def __init__(self,
               input_width,
               start_n_top,
               end_n_top,
               activation='tanh',
               dropout_rate=0.,
               initializer='glorot_uniform',
               **kwargs):
    super().__init__(**kwargs)
    self._config = {
        'input_width': input_width,
        'activation': activation,
        'initializer': initializer,
        'start_n_top': start_n_top,
        'end_n_top': end_n_top,
        'dropout_rate': dropout_rate,
    }
    self._start_n_top = start_n_top
    self._end_n_top = end_n_top
    self.start_logits_dense = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=initializer,
        name='predictions/transform/start_logits')

    self.end_logits_inner_dense = tf.keras.layers.Dense(
        units=input_width,
        kernel_initializer=initializer,
        activation=activation,
        name='predictions/transform/end_logits/inner')
    self.end_logits_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12,
        name='predictions/transform/end_logits/layernorm')
    self.end_logits_output_dense = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=initializer,
        name='predictions/transform/end_logits/output')

    self.answer_logits_inner = tf.keras.layers.Dense(
        units=input_width,
        kernel_initializer=initializer,
        activation=activation,
        name='predictions/transform/answer_logits/inner')
    self.answer_logits_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
    self.answer_logits_output = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=initializer,
        use_bias=False,
        name='predictions/transform/answer_logits/output')

  def end_logits(self, inputs):
    """Computes the end logits.

    Input shapes into the inner, layer norm, output layers should match.

    During training, inputs shape should be
    [batch_size, seq_length, input_width].

    During inference, input shapes should be
    [batch_size, seq_length, start_n_top, input_width].

    Args:
      inputs: The input for end logits.

    Returns:
      Calculated end logits.

    """
    if len(tf.shape(inputs)) == 3:
      # inputs: [B, S, H] -> [B, S, 1, H]
      inputs = tf.expand_dims(inputs, axis=2)

    end_logits = self.end_logits_inner_dense(inputs)
    end_logits = self.end_logits_layer_norm(end_logits)
    end_logits = self.end_logits_output_dense(end_logits)
    end_logits = tf.squeeze(end_logits)
    if tf.rank(end_logits) > 2:
      # shape = [B, S, K] -> [B, K, S]
      end_logits = tf.transpose(end_logits, [0, 2, 1])

    return end_logits

  def call(self,
           sequence_data,
           class_index,
           position_mask=None,
           start_positions=None,
           training=False):
    """Implements call().

    Einsum glossary:
    - b: the batch size.
    - l: the sequence length.
    - h: the hidden size, or input width.
    - k: the start/end top n.

    Args:
      sequence_data: The input sequence data of shape
        (batch_size, seq_length, input_width).
      class_index: The class indices of the inputs of shape (batch_size,).
      position_mask: Invalid position mask such as query and special symbols
        (e.g. PAD, SEP, CLS) of shape (batch_size,).
      start_positions: The start positions of each example of shape
        (batch_size,).
      training: Whether or not this is the training phase.

    Returns:
      A dictionary with the keys 'cls_logits' and
        - (if training)              'start_log_probs', 'end_log_probs'.
        - (if inference/beam search) 'start_top_log_probs', 'start_top_index',
                                     'end_top_log_probs', 'end_top_index'.

    """
    seq_length = tf.shape(sequence_data)[1]
    start_logits = self.start_logits_dense(sequence_data)
    start_logits = tf.squeeze(start_logits, -1)
    start_log_probs, masked_start_logits = _apply_position_mask(
        start_logits, position_mask)

    compute_with_beam_search = not training or start_positions is None

    if compute_with_beam_search:
      # Compute end logits using beam search.
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=self._start_n_top)
      start_index = tf.one_hot(
          start_top_index, depth=seq_length, axis=-1, dtype=tf.float32)
      # start_index: [batch_size, end_n_top, seq_length]

      start_features = tf.einsum('blh,bkl->bkh', sequence_data, start_index)
      start_features = tf.tile(start_features[:, None, :, :],
                               [1, seq_length, 1, 1])
      # start_features: [batch_size, seq_length, end_n_top, input_width]

      end_input = tf.tile(sequence_data[:, :, None],
                          [1, 1, self._start_n_top, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      # end_input: [batch_size, seq_length, end_n_top, 2*input_width]
    else:
      start_positions = tf.reshape(start_positions, -1)
      start_index = tf.one_hot(
          start_positions, depth=seq_length, axis=-1, dtype=tf.float32)
      # start_index: [batch_size, seq_length]

      start_features = tf.einsum('blh,bl->bh', sequence_data, start_index)
      start_features = tf.tile(start_features[:, None, :], [1, seq_length, 1])
      # start_features: [batch_size, seq_length, input_width]

      end_input = tf.concat([sequence_data, start_features],
                            axis=-1)
      # end_input: [batch_size, seq_length, 2*input_width]

    end_logits = self.end_logits(end_input)
    end_log_probs, _ = _apply_position_mask(end_logits, position_mask)

    output_dict = {}
    if training:
      output_dict['start_log_probs'] = start_log_probs
      output_dict['end_log_probs'] = end_log_probs
    else:
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=self._end_n_top)
      end_top_log_probs = tf.reshape(end_top_log_probs,
                                     [-1, self._start_n_top * self._end_n_top])
      end_top_index = tf.reshape(end_top_index,
                                 [-1, self._start_n_top * self._end_n_top])
      output_dict['start_top_log_probs'] = start_top_log_probs
      output_dict['start_top_index'] = start_top_index
      output_dict['end_top_log_probs'] = end_top_log_probs
      output_dict['end_top_index'] = end_top_index

    # get the representation of CLS
    class_index = tf.one_hot(class_index, seq_length, axis=-1, dtype=tf.float32)
    class_feature = tf.einsum('blh,bl->bh', sequence_data, class_index)

    # get the representation of START
    start_p = tf.nn.softmax(masked_start_logits, axis=-1)
    start_feature = tf.einsum('blh,bl->bh', sequence_data, start_p)

    answer_feature = tf.concat([start_feature, class_feature], -1)
    answer_feature = self.answer_logits_inner(answer_feature)
    answer_feature = self.answer_logits_dropout(answer_feature)
    class_logits = self.answer_logits_output(answer_feature)
    class_logits = tf.squeeze(class_logits, -1)
    output_dict['class_logits'] = class_logits
    return output_dict

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
