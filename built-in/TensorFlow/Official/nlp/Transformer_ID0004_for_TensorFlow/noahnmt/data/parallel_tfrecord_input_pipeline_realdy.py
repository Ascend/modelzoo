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
class ParallelTfrecordInputPipelineRealdy(InputPipeline):
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
        "fix_batch": False,
        "max_length": 128,
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
      # tf.logging.info(t)
      t = tf.sparse.to_dense(t)
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      # tf.logging.info(t.get_shape().as_list())
      # assert t.get_shape().as_list()[0] is not None
      example[name] = t
    
    del example["source_sos_ids"]
    del example["source_sos_mask"]

    return example
 

  def read_data(self, seed=None, **kwargs):
    input_files = self.params["files"].split(",")
    num_threads = self.params["num_threads"]
    
    batch_size = self.params["batch_size"]
    max_length= self.params["max_length"]
    assert not self.params["fix_batch"]

    name_to_features = {
      "source_sos_ids":
        tf.VarLenFeature( tf.int64),
      "source_sos_mask":
        tf.VarLenFeature( tf.int64),
      "source_eos_ids":
        tf.VarLenFeature( tf.int64),
      "source_eos_mask":
        tf.VarLenFeature( tf.int64),
      "target_sos_ids":
        tf.VarLenFeature( tf.int64),
      "target_sos_mask":
        tf.VarLenFeature( tf.int64),
      "target_eos_ids":
        tf.VarLenFeature( tf.int64),
      "target_eos_mask":
        tf.VarLenFeature( tf.int64),
    }

    #
    d = tf.data.TFRecordDataset(input_files)

    # for multi-device training
    if self.params["num_shards"] > 1:
      d = d.shard(
          num_shards=self.params["num_shards"],
          index=self.params["shard_index"])
    
    # decode tfrecord
    d = d.map(lambda record: self._decode_record(record, name_to_features),
              num_parallel_calls=num_threads)

    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    assert  self.mode == tf.estimator.ModeKeys.TRAIN
    d = d.shuffle(100000)
    d = d.repeat()


    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(sent_num, x):
      # The first three entries are the source and target line rows;
      # these have unknown-length vectors.  The last two entries are
      # the source and target row sizes; these are scalars.
      padded_shapes = {
          "source_eos_ids": tf.TensorShape([None]),
          "source_eos_mask": tf.TensorShape([None]),
          "target_sos_ids": tf.TensorShape([None]),
          "target_sos_mask": tf.TensorShape([None]),
          "target_eos_ids": tf.TensorShape([None]),
          "target_eos_mask": tf.TensorShape([None])}

      return x.padded_batch(
          sent_num,
          padded_shapes=padded_shapes,
          padding_values=None)


    # boundary for bucketing
    x = 8
    boundaries = []
    while x < max_length:
      boundaries.append(x)
      x = x + 8
    boundaries.append(max_length)
    
    
    # dynamic batch size
    batch_sizes = [max(1, batch_size // length) for length in boundaries]
    

    def key_func(dict_):
      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      src_len = tf.size(dict_["source_eos_ids"])
      tgt_len = tf.size(dict_["target_eos_ids"])

      seq_len = tf.maximum(src_len, tgt_len)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = tf.logical_and(
          tf.less(buckets_min, seq_len),
          tf.less_equal(seq_len, buckets_max))
      bucket_id = tf.reduce_min(tf.where(conditions_c))

      return bucket_id


    def batch_size_func(bucket_id):
      if isinstance(batch_sizes, list):
        batch_sizes_tensor = tf.constant(batch_sizes, dtype=tf.int64)
        return batch_sizes_tensor[bucket_id]
      return batch_sizes


    def reduce_func(bucket_id, windowed_data):
      return batching_func(batch_size_func(bucket_id), windowed_data)


    def window_size_func(bucket_id):
      return batch_size_func(bucket_id)
    
    

    batched_dataset = d.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, 
            window_size=None, window_size_func=window_size_func)
    )

    batched_dataset = batched_dataset.prefetch(buffer_size=100)
    inputs = batched_dataset.make_one_shot_iterator().get_next()

    # rename
    features = {}
    features["source_ids"] = inputs["source_eos_ids"]
    features["source_mask"] = inputs["source_eos_mask"]
    features["target_ids"] = inputs["target_sos_ids"]
    features["target_mask"] = inputs["target_sos_mask"]
    features["label_ids"] = inputs["target_eos_ids"]
    features["label_weight"] = inputs["target_eos_mask"]
    # features["batch_size"] = inputs["batch_size"]
    # features["seq_length"] = inputs["seq_length"]

    # for name in features:
    #   tensor = features[name]
    #   shape_list = tensor.get_shape().as_list()
    #   tf.logging.info(shape_list)
    #   for s in shape_list:
    #     assert s is not None
    #     assert isinstance(s, int)

    # for name in features:
    #   features[name].set_shape([None, None])

    return features
