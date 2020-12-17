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

from noahnmt.metrics.metric_specs import accumulate_strings, TextMetric


@registry.register_class("multikd_bleu")
class MultikdBleuMetric(TextMetric):
  """Calculates BLEU score using the Moses multi-bleu.perl script.
  """
  
  def __init__(self, params):
    super(MultikdBleuMetric, self).__init__(params, "bleu")


  @staticmethod
  def default_params():
    params = TextMetric.default_params()
    params.update({
        "lang_pairs": ""
    })
    return params


  def create_metric_ops(self, predictions):
    """Creates (value, update_op) tensors
    """
    self.lang_pairs = self.params["lang_pairs"].split(",")

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

      # language pair info
      lang_pair = tf.tile(
          input=tf.expand_dims(predictions["lang_pair"],0),
          multiples=[tf.shape(predictions["predicted_tokens"])[0]])
      lang_pair_value, lang_pair_update = accumulate_strings(
          values=lang_pair, name="lang_pairs")

      metric_value = tf.py_func(
          func=self._py_func,
          inp=[sources_value, targets_value, lang_pair_value],
          Tout=constant_utils.DT_FLOAT(),
          name="value")

    # with tf.control_dependencies([sources_update, targets_update]):
    #   update_op = tf.identity(metric_value, name="update_op")
    update_op = tf.group(sources_update.op, targets_update.op, lang_pair_update)

    return metric_value, update_op


  def _py_func(self, hypotheses, references, lang_pairs):
    """Wrapper function that converts tensors to unicode and slices
      them until the EOS token is found.
    """
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
      hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
      references = np.char.encode(references, "utf-8")
    if lang_pairs.dtype.kind == np.dtype("U"):
      lang_pairs = np.char.encode(lang_pairs, "utf-8")

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

    return self.metric_fn(sliced_hypotheses, sliced_references, lang_pairs) #pylint: disable=E1102


  def metric_fn(self, hypotheses, references, lang_pairs):
    # byte to str
    lang_pairs = [x.decode('utf-8') for x in lang_pairs]

    # convenient func
    def _calc_bleu(hyp, ref):
      return multi_bleu.moses_multi_bleu(
            hyp, 
            ref, 
            lowercase=False,
            script=self.params["script"])
    
    # total bleu
    results = [_calc_bleu(hypotheses, references)]

    for lang_pair in self.lang_pairs:
      # filter according to each language pair
      filtered = [(hyp, ref) for lp, hyp, ref in zip(lang_pairs, hypotheses, references) 
                                if lp == lang_pair]
      if len(filtered) > 0:
        hypotheses_, references_ = zip(*filtered)
        results.append(_calc_bleu(hypotheses_, references_))
      else:
        raise ValueError("No data form language pair: %s" % lang_pair)
        # results.append(0.)
    return np.asarray(results)
