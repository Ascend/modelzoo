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
# from noahnmt.utils import align_utils
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.layers import common_layers as common_utils


@registry.register_class
class ParallelTfrecordInputPipeline(InputPipeline):
  """An input pipeline that reads two parallel (line-by-line aligned) text
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
        "files": "",
        "max_length": None,
        "fix_batch": True,
        "shuffle": True,
    })
    return params
  

  def _decode_record(self, record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example
 

  def read_data(self, seed=None, **kwargs):
    input_files = self.params["files"].split(",")
    max_seq_length = self.params["max_length"]
    batch_size = self.params["batch_size"]
    num_threads = self.params["num_threads"]

    name_to_features = {
      "source_sos_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "source_sos_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "source_eos_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "source_eos_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "target_sos_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "target_sos_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "target_eos_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
      "target_eos_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    
    d = tf.data.TFRecordDataset(input_files)

    if self.params["num_shards"] > 1:
     d = d.shard(
         num_shards=self.params["num_shards"],
         index=self.params["shard_index"])

    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    if  self.mode == tf.estimator.ModeKeys.TRAIN:
      d  = d.shuffle(1000*batch_size)
      d = d.repeat()
    #else:
    #  d = d.take(8)
  
    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
      tf.contrib.data.map_and_batch(
        lambda record: self._decode_record(record, name_to_features),
        batch_size=batch_size,
        num_parallel_batches=num_threads,
        drop_remainder=True))
    d = d.prefetch(buffer_size=100)
    inputs = d.make_one_shot_iterator().get_next()

    # rename
    features = {}
    features["source_ids"] = inputs["source_eos_ids"]
    features["source_mask"] = inputs["source_eos_mask"]
    features["target_ids"] = inputs["target_sos_ids"]
    features["target_mask"] = inputs["target_sos_mask"]
    features["label_ids"] = inputs["target_eos_ids"]
    features["label_weight"] = inputs["target_eos_mask"]

    return features
