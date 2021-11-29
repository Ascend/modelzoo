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

"""Collection of MetricSpecs for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate
import abc
import numpy as np
import six

import tensorflow as tf

from noahnmt.configurable import Configurable
from noahnmt.data import postproc
from noahnmt.metrics import multi_bleu
from noahnmt.utils import vocab_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import registry


def accumulate_strings(values, name="strings"):
  """Accumulates strings into a vector.

  Args:
    values: A 1-d string tensor that contains values to add to the accumulator.

  Returns:
    A tuple (value_tensor, update_op).
  """
  tf.assert_type(values, tf.string)
  strings = tf.Variable(
      name=name,
      initial_value=[],
      dtype=tf.string,
      trainable=False,
      collections=[],
      validate_shape=True)
  value_tensor = tf.identity(strings)
  update_op = tf.assign(
      ref=strings, value=tf.concat([strings, values], 0), validate_shape=False)
  return value_tensor, update_op


@six.add_metaclass(abc.ABCMeta)
class TextMetric(Configurable):
  """Abstract class for text-based metrics calculated based on
  hypotheses and references. Subclasses must implement `metric_fn`.

  Args:
    name: A name for the metric
    separator: A separator used to join predicted tokens. Default to space.
    eos_token: A string token used to find the end of a sequence. Hypotheses
      and references will be slcied until this token is found.
  """

  def __init__(self, params, name):
    Configurable.__init__(self, params, tf.estimator.ModeKeys.EVAL)
    self._name = name
    self._eos_token = self.params["eos_token"]
    self._sos_token = self.params["sos_token"]
    self._separator = self.params["separator"]
    self._postproc_fn = None
    if self.params["postproc_fn"]:
      self._postproc_fn = postproc.get_postproc_fn(self.params["postproc_fn"])

  @property
  def name(self):
    """Name of the metric"""
    return self._name

  @staticmethod
  def default_params():
    return {
        "sos_token": vocab_utils.SOS,
        "eos_token": vocab_utils.EOS,
        "separator": " ",
        "postproc_fn": "",
        "script": "",
    }

  def create_metric_ops(self, predictions):
    """Creates (value, update_op) tensors
    """
    with tf.variable_scope(self._name):
      # Join tokens into single strings
      predictions_flat = tf.reduce_join(
          predictions["predicted_tokens"], 1, separator=self._separator)
      labels_flat = tf.reduce_join(
          predictions["target_tokens"][:,1:], 1, separator=self._separator)

      sources_value, sources_update = accumulate_strings(
          values=predictions_flat, name="sources")
      targets_value, targets_update = accumulate_strings(
          values=labels_flat, name="targets")

      metric_value = tf.py_func(
          func=self._py_func,
          inp=[sources_value, targets_value],
          Tout=constant_utils.DT_FLOAT(),
          name="value")

    # with tf.control_dependencies([sources_update, targets_update]):
    #   update_op = tf.identity(metric_value, name="update_op")
    update_op = tf.group(sources_update.op, targets_update.op)

    return metric_value, update_op


  def _py_func(self, hypotheses, references):
    """Wrapper function that converts tensors to unicode and slices
      them until the EOS token is found.
    """
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
      hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
      references = np.char.encode(references, "utf-8")

    # Convert back to unicode object
    hypotheses = [_.decode("utf-8") for _ in hypotheses]
    references = [_.decode("utf-8") for _ in references]

    # Slice all hypotheses and references up to SOS -> EOS
    sliced_hypotheses = [postproc.slice_text(
        _, self._eos_token, self._sos_token) for _ in hypotheses]
    sliced_references = [postproc.slice_text(
        _, self._eos_token, self._sos_token) for _ in references]

    # Apply postprocessing function
    if self._postproc_fn:
      sliced_hypotheses = [self._postproc_fn(_) for _ in sliced_hypotheses]
      sliced_references = [self._postproc_fn(_) for _ in sliced_references]

    return self.metric_fn(sliced_hypotheses, sliced_references) #pylint: disable=E1102


  def metric_fn(self, hypotheses, references):
    """Calculates the value of the metric.

    Args:
      hypotheses: A python list of strings, each corresponding to a
        single hypothesis/example.
      references: A python list of strings, each corresponds to a single
        reference. Must have the same number of elements of `hypotheses`.

    Returns:
      A float value.
    """
    raise NotImplementedError()


@registry.register_class("bleu")
class BleuMetric(TextMetric):
  """Calculates BLEU score using the Moses multi-bleu.perl script.
  """
  
  def __init__(self, params):
    super(BleuMetric, self).__init__(params, "bleu")


  def metric_fn(self, hypotheses, references):
    return multi_bleu.moses_multi_bleu(
        hypotheses, 
        references, 
        lowercase=False,
        script=self.params["script"])



@registry.register_class("perplexity")
class PerplexityMetric(Configurable):
  """A MetricSpec to calculate straming perplexity"""

  def __init__(self, params):
    """Initializer"""
    Configurable.__init__(self, params, tf.estimator.ModeKeys.EVAL)

  @staticmethod
  def default_params():
    return {}

  @property
  def name(self):
    """Name of the metric"""
    return "perplexity"


  def create_metric_ops(self, predictions):
    """Creates the metric op"""
    if "crossent" in predictions:
      loss_mask = tf.sequence_mask(
          lengths=tf.to_int32(predictions["target_len"]),
          maxlen=tf.to_int32(tf.shape(predictions["crossent"])[1]))
      value, update_op =  tf.metrics.mean(predictions["crossent"], loss_mask)
      return tf.exp(value), update_op
    else:
      return tf.constant(1.0), tf.no_op()
