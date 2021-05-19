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
"""BERT models that are compatible with TF 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf2_common.modeling import tf_utils
import configs
from modeling import losses
from modeling import networks
from modeling.networks import bert_classifier
from modeling.networks import bert_pretrainer
from modeling.networks import bert_span_labeler
from modeling.layers import bert_dropout
from metrics_sparse_int32 import sparse_categorical_accuracy_int32


class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, vocab_size, **kwargs):
    super(BertPretrainLossAndMetricLayer, self).__init__(**kwargs)
    self._vocab_size = vocab_size
    self.config = {
        'vocab_size': vocab_size,
    }

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_example_loss, sentence_output, sentence_labels,
                   next_sentence_loss):
    """Adds metrics."""
    #masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
    #    lm_labels, lm_output)
    masked_lm_accuracy = sparse_categorical_accuracy_int32(lm_labels, lm_output)
    numerator = tf.reduce_sum(masked_lm_accuracy * lm_label_weights)
    denominator = tf.reduce_sum(lm_label_weights)
    masked_lm_accuracy = numerator / denominator
    masked_lm_sum_loss = tf.reduce_sum(lm_example_loss * lm_label_weights)
    # self.add_metric(
    #     masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')
    # self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    #next_sentence_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
    #    sentence_labels, sentence_output)
    next_sentence_accuracy = sparse_categorical_accuracy_int32(sentence_labels, sentence_output)
    next_sentence_num = tf.reduce_sum(next_sentence_accuracy)
    next_sentence_denom = tf.size(next_sentence_accuracy)
    # self.add_metric(
    #     next_sentence_accuracy,
    #     name='next_sentence_accuracy',
    #     aggregation='mean')
    # self.add_metric(
    #     next_sentence_loss, name='next_sentence_loss', aggregation='mean')
    other_outputs = dict(
      masked_lm_num = numerator,
      masked_lm_denom = denominator,
      masked_lm_accuracy = masked_lm_accuracy,
      lm_example_loss = lm_example_loss,
      masked_lm_sum_loss = masked_lm_sum_loss,
      next_sentence_accuracy = next_sentence_accuracy,
      next_sentence_loss = next_sentence_loss,
      next_sentence_num = next_sentence_num,
      next_sentence_denom = next_sentence_denom)
    return other_outputs


  def call(self, lm_output, sentence_output, lm_label_ids, lm_label_weights,
           sentence_labels):
    """Implements call() for the layer."""
    lm_label_weights = tf.cast(lm_label_weights, tf.float32)
    lm_output = tf.cast(lm_output, tf.float32)
    sentence_output = tf.cast(sentence_output, tf.float32)

    mask_label_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=lm_label_ids, predictions=lm_output, weights=lm_label_weights)
    sentence_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=sentence_labels, predictions=sentence_output)
    loss = mask_label_loss + sentence_loss
    batch_shape = tf.slice(tf.shape(sentence_labels), [0], [1])
    # TODO(hongkuny): Avoids the hack and switches add_loss.
    final_loss = tf.fill(batch_shape, loss)

    # TODO(b/122840926): metrics use distribution strategy merge_call() and do
    # not work with tf.function(compile=True). Either fix this issue or move
    # metric aggregation outside the model.
    metric_outputs = self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                     mask_label_loss, sentence_output, sentence_labels,
                     sentence_loss)
    return final_loss, metric_outputs


def get_transformer_encoder(bert_config, sequence_length):
  """Gets a 'TransformerEncoder' object.

  Args:
    bert_config: A 'modeling.BertConfig' or 'modeling.AlbertConfig' object.
    sequence_length: Maximum sequence length of the training data.

  Returns:
    A networks.TransformerEncoder object.
  """
  kwargs = dict(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      sequence_length=sequence_length,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))
  assert isinstance(bert_config, configs.BertConfig)
  return networks.TransformerEncoder(**kwargs)


def pretrain_model(bert_config,
                   seq_length,
                   max_predictions_per_seq,
                   initializer=None):
  """Returns model to be used for pre-training.

  Args:
      bert_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of tokens in sequence to mask out
        and use for pretraining.
      initializer: Initializer for weights in BertPretrainer.

  Returns:
      Pretraining model as well as core BERT submodel from which to save
      weights after pretraining.
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
  masked_lm_positions = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_positions',
      dtype=tf.int32)
  masked_lm_ids = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
  masked_lm_weights = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_weights',
      dtype=tf.int32)
  next_sentence_labels = tf.keras.layers.Input(
      shape=(1,), name='next_sentence_labels', dtype=tf.int32)

  transformer_encoder = get_transformer_encoder(bert_config, seq_length)
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)
  pretrainer_model = bert_pretrainer.BertPretrainer(
      network=transformer_encoder,
      num_classes=2,  # The next sentence prediction label has two classes.
      num_token_predictions=max_predictions_per_seq,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      initializer=initializer,
      output='predictions')

  lm_output, sentence_output = pretrainer_model(
      [input_word_ids, input_mask, input_type_ids, masked_lm_positions])

  pretrain_loss_layer = BertPretrainLossAndMetricLayer(
      vocab_size=bert_config.vocab_size)
  output_loss = pretrain_loss_layer(lm_output, sentence_output, masked_lm_ids,
                                    masked_lm_weights, next_sentence_labels)
  keras_model = tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids,
          'masked_lm_positions': masked_lm_positions,
          'masked_lm_ids': masked_lm_ids,
          'masked_lm_weights': masked_lm_weights,
          'next_sentence_labels': next_sentence_labels,
      },
      outputs=output_loss)
  return keras_model, transformer_encoder, pretrainer_model


def squad_model(bert_config,
                max_seq_length,
                initializer=None,
                hub_module_url=None,
                hub_module_trainable=True):
  """Returns BERT Squad model along with core BERT model to import weights.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    max_seq_length: integer, the maximum input sequence length.
    initializer: Initializer for the final dense layer in the span labeler.
      Defaulted to TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.

  Returns:
    A tuple of (1) keras model that outputs start logits and end logits and
    (2) the core BERT transformer encoder.
  """
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)
  if not hub_module_url:
    bert_encoder = get_transformer_encoder(bert_config, max_seq_length)
    return bert_span_labeler.BertSpanLabeler(
        network=bert_encoder, initializer=initializer), bert_encoder

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
  core_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
  pooled_output, sequence_output = core_model(
      [input_word_ids, input_mask, input_type_ids])
  bert_encoder = tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids,
      },
      outputs=[sequence_output, pooled_output],
      name='core_model')
  return bert_span_labeler.BertSpanLabeler(
      network=bert_encoder, initializer=initializer), bert_encoder


def classifier_model(bert_config,
                     num_labels,
                     max_seq_length,
                     final_layer_initializer=None,
                     hub_module_url=None,
                     hub_module_trainable=True):
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    bert_config: BertConfig or AlbertConfig, the config defines the core BERT or
      ALBERT model.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    final_layer_initializer: Initializer for final dense layer. Defaulted
      TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  if final_layer_initializer is not None:
    initializer = final_layer_initializer
  else:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  if not hub_module_url:
    bert_encoder = get_transformer_encoder(bert_config, max_seq_length)
    return bert_classifier.BertClassifier(
        bert_encoder,
        num_classes=num_labels,
        dropout_rate=bert_config.hidden_dropout_prob,
        initializer=initializer), bert_encoder

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
  bert_model = hub.KerasLayer(
      hub_module_url, trainable=hub_module_trainable)
  pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
  #output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
  #    pooled_output)
  output = bert_dropout.Dropout(rate=bert_config.hidden_dropout_prob)(pooled_output)

  output = tf.keras.layers.Dense(
      num_labels, kernel_initializer=initializer, name='output')(
          output)
  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs=output), bert_model
