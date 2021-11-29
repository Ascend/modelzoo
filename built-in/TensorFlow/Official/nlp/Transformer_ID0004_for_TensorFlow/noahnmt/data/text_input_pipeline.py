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
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys
import six
import math
import numpy as np

import tensorflow as tf

from noahnmt.data.input_pipeline import InputPipeline
from noahnmt.utils import constant_utils
from noahnmt.utils import registry


@registry.register_class
class TextInputPipeline(InputPipeline):
  """An input pipeline that reads text
  files.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "source_files": "",
        "source_max_len": None,
        "source_delimiter": " ",
        "source_reverse": False,
    })
    return params


  def _create_batched_dataset(self, src_dataset, seed=None, **kwargs):
    src_max_len = self.params["source_max_len"]
    batch_size = self.params["batch_size"] * self.params["batch_multiplier"]
    num_threads = self.params["num_threads"]
    output_buffer_size = self.params["output_buffer_size"]
    sos = self.params["sos"]
    eos = self.params["eos"]

    if output_buffer_size is None:
      output_buffer_size = 1000 * batch_size

    # convert to dict
    src_dataset = src_dataset.map(
        lambda src: {"source_tokens": src},
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    if self.params["num_shards"] > 1:
      src_dataset = src_dataset.shard(
          num_shards=self.params["num_shards"],
          index=self.params["shard_index"])

    src_dataset = src_dataset.map(
      lambda dict_: dict_.update({
        "source_tokens": tf.string_split([dict_["source_tokens"]]).values
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_dataset = src_dataset.filter(
      lambda dict_: tf.size(dict_["source_tokens"]) > 0)

    if src_max_len:
      src_dataset = src_dataset.filter(
        lambda dict_: tf.size(dict_["source_tokens"]) <= src_max_len)
    
    if self.params["source_reverse"]:
      src_dataset = src_dataset.map(
        lambda dict_: dict_.update({
            "source_tokens": tf.reverse(dict_["source_tokens"], axis=[0])
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    src_dataset = src_dataset.map(
      lambda dict_: dict_.update({
          "source_tokens": tf.concat((dict_["source_tokens"], [eos]), 0)
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)
    
    if self.params["source_sos"]:
      src_dataset = src_dataset.map(
        lambda dict_: dict_.update({
            "source_tokens": tf.concat(([sos], dict_["source_tokens"]), 0)
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Add in the word counts.
    src_dataset = src_dataset.map(
      lambda dict_: dict_.update({
        "source_len": tf.size(dict_["source_tokens"], 
                              out_type=constant_utils.DT_INT())
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
      padded_shapes = {
          "source_tokens": tf.TensorShape([None]), 
          "source_len": tf.TensorShape([])}
      padding_values = {
          "source_tokens": eos, 
          "source_len": 0}

      return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=padded_shapes,  # src_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=padding_values)  # tgt_len -- unused

    return batching_func(src_dataset)


  def read_data(self, seed=None, **kwargs):
    src_dataset = tf.data.TextLineDataset(self.params["source_files"].split(","))
    batched_dataset = self._create_batched_dataset(src_dataset, seed=seed, **kwargs)
    batched_dataset = batched_dataset.prefetch(1000)
    batched_iter = batched_dataset.make_one_shot_iterator()

    # return features
    features = batched_iter.get_next()
    return features
