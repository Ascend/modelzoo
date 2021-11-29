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
class ParallelTextInputPipeline(InputPipeline):
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
        "source_files": "",
        "target_files": "",
        "align_file": None,
        "kd_ref_file": None,
        "source_max_len": None,
        "target_max_len": None,
        "filter_long_sents": True,
        "source_delimiter": " ",
        "target_delimiter": " ",
        "fix_batch": True,
        "source_reverse": False,
        "shuffle": True,
        "pad_to_eight": False,
    })
    return params
  

  def _create_batched_dataset(self, src_tgt_dataset, seed=None, **kwargs):
    src_max_len = self.params["source_max_len"]
    tgt_max_len = self.params["target_max_len"]
    batch_size = self.params["batch_size"]
    num_threads = self.params["num_threads"]
    output_buffer_size = self.params["output_buffer_size"]
    eos = self.params["eos"]
    sos = self.params["sos"]

    if output_buffer_size is None:
      output_buffer_size = 1000 * batch_size * self.params["batch_multiplier"]
      if not self.params["fix_batch"]:
        # in this case, batch_size is usually a large value
        # divide by an approximate average sentence length
        output_buffer_size //= 20

    # use dict for convenience
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: {"source_tokens":src, "target_tokens": tgt}, 
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    align_file = self.params["align_file"]
    # if align_file:
    #   assert self.mode == tf.estimator.ModeKeys.TRAIN
    #   align_dataset = align_utils.get_align_dataset(align_file)
    #   src_tgt_dataset = tf.data.Dataset.zip((src_tgt_dataset, align_dataset))
    #   src_tgt_dataset = src_tgt_dataset.map(
    #     lambda dict_, align: dict_.update({
    #         "align_indices": align[0],
    #         "align_values": align[1]}) or dict_, 
    #     num_parallel_calls=num_threads).prefetch(output_buffer_size)
      
    #   # filter out empty-align lines, only contain </s>-</s>
    #   src_tgt_dataset = src_tgt_dataset.filter(
    #     lambda dict_: tf.size(dict_["align_values"]) > 1)
    
    kd_ref_file = self.params["kd_ref_file"]
    if kd_ref_file:
      kd_dataset = tf.data.TextLineDataset(kd_ref_file)
      src_tgt_dataset = tf.data.Dataset.zip((src_tgt_dataset, kd_dataset))
      src_tgt_dataset = src_tgt_dataset.map(
          lambda dict_, kd: dict_.update({
              "kd_ref_tokens": kd}) or dict_, 
          num_parallel_calls=num_threads).prefetch(output_buffer_size)

    if self.params["num_shards"] > 1 and self.mode == tf.estimator.ModeKeys.TRAIN:
      src_tgt_dataset = src_tgt_dataset.shard(
          num_shards=self.params["num_shards"],
          index=self.params["shard_index"])
    
    src_tgt_dataset = src_tgt_dataset.map(
      lambda dict_: dict_.update({
        "source_tokens": tf.string_split([dict_["source_tokens"]]).values, # src
        "target_tokens": tf.string_split([dict_["target_tokens"]]).values, # tgt
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)
    
    if kd_ref_file:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda dict_: dict_.update({
          "kd_ref_tokens": tf.string_split([dict_["kd_ref_tokens"]]).values, # tgt
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda dict_: tf.logical_and(
          tf.size(dict_["source_tokens"]) > 0, 
          tf.size(dict_["target_tokens"]) > 0))
    
    if kd_ref_file:
      src_tgt_dataset = src_tgt_dataset.filter(
        lambda dict_: tf.size(dict_["kd_ref_tokens"]) > 0)
    
    
    if self.mode == tf.estimator.ModeKeys.TRAIN and self.params["shuffle"]:
      src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, seed)
      # num_epochs loops before batching to avoid frequent small batches
      # after shuffle to avoid repeat sentences in the same batch
      src_tgt_dataset = src_tgt_dataset.repeat()

    if src_max_len:
      if self.params["filter_long_sents"]:
        src_tgt_dataset = src_tgt_dataset.filter(
          lambda dict_: tf.size(dict_["source_tokens"]) <= src_max_len)
      else:
        src_tgt_dataset = src_tgt_dataset.map(
          lambda dict_: dict_.update({
              "source_tokens": dict_["source_tokens"][:src_max_len]
          }) or dict_,
          num_parallel_calls=num_threads).prefetch(output_buffer_size)
    if tgt_max_len:
      if self.params["filter_long_sents"]:
        src_tgt_dataset = src_tgt_dataset.filter(
          lambda dict_: tf.size(dict_["target_tokens"]) <= tgt_max_len)
        if kd_ref_file:
          src_tgt_dataset = src_tgt_dataset.filter(
            lambda dict_: tf.size(dict_["kd_ref_tokens"]) <= tgt_max_len)
      else:
        src_tgt_dataset = src_tgt_dataset.map(
          lambda dict_: dict_.update({
              "target_tokens": dict_["target_tokens"][:tgt_max_len]
          }) or dict_,
          num_parallel_calls=num_threads).prefetch(output_buffer_size)
        if kd_ref_file:
          raise NotImplementedError("kd_ref_file")

    if self.params["source_reverse"]:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda dict_: dict_.update({
            "source_tokens": tf.reverse(dict_["source_tokens"], axis=[0])
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Create a tgt prefixed with <sos> and suffixed with <eos>.
    # src suffixed with <eos>
    src_tgt_dataset = src_tgt_dataset.map(
      lambda dict_: dict_.update({
          "source_tokens": tf.concat((dict_["source_tokens"], [eos]), 0),
          "target_tokens": tf.concat(([sos], dict_["target_tokens"], [eos]), 0),
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)
    
    if self.params["source_sos"]:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda dict_: dict_.update({
            "source_tokens": tf.concat(([sos], dict_["source_tokens"]), 0),
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)
    
    if kd_ref_file:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda dict_: dict_.update({
            "kd_ref_tokens": tf.concat(([sos], dict_["kd_ref_tokens"], [eos]), 0),
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Add in the word counts.  Subtract one from the target to avoid counting
    # the target_input <eos> tag (resp. target_output <sos> tag).
    src_tgt_dataset = src_tgt_dataset.map(
      lambda dict_: dict_.update({
        "source_len": tf.size(dict_["source_tokens"]), 
        "target_len": tf.size(dict_["target_tokens"])-1
      }) or dict_,
      num_parallel_calls=num_threads).prefetch(output_buffer_size)
    
    if kd_ref_file:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda dict_: dict_.update({
          "kd_ref_len": tf.size(dict_["kd_ref_tokens"])-1
        }) or dict_,
        num_parallel_calls=num_threads).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(batch_size, x):
      # The first three entries are the source and target line rows;
      # these have unknown-length vectors.  The last two entries are
      # the source and target row sizes; these are scalars.
      padded_shapes = {
          "source_tokens": tf.TensorShape([None]),  # src
          "target_tokens": tf.TensorShape([None]),  # tgt
          "source_len": tf.TensorShape([]),  # src_len
          "target_len": tf.TensorShape([])} # tgt_len
      # Pad the source and target sequences with eos tokens.
      # (Though notice we don't generally need to do this since
      # later on we will be masking out calculations past the true sequence.
      padded_values = {
          "source_tokens": eos,  "target_tokens": eos,  
          "source_len": 0,  "target_len": 0}

      if align_file:
        padded_shapes.update({
            "align_indices": tf.TensorShape([None, 2]), 
            "align_values": tf.TensorShape([None])}) # align
        padded_values.update({
            "align_indices": 0, 
            "align_values": tf.constant(0., dtype=constant_utils.DT_FLOAT())})
      
      if kd_ref_file:
        padded_shapes.update({
            "kd_ref_tokens": tf.TensorShape([None]), 
            "kd_ref_len": tf.TensorShape([])}) # align
        padded_values.update({
            "kd_ref_tokens": eos, 
            "kd_ref_len": 0})
        
      return x.padded_batch(
          batch_size,
          padded_shapes=padded_shapes,
          padding_values=padded_values)

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # Calculate boundaries
      min_length = 8
      max_length = src_max_len if src_max_len else 1024
      if tgt_max_len:
        max_length = max(max_length, tgt_max_len)

      # boundary for bucketing
      x = min_length
      boundaries = []
      while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * 1.2))
      
      if not self.params["fix_batch"]:
        batch_sizes = [max(1, batch_size // length)
                        for length in boundaries + [max_length]]
        if self.params["pad_to_eight"]:
          batch_sizes = [max(1, x//8*8) for x in batch_sizes]
        batch_sizes = [b * self.params["batch_multiplier"] for b in batch_sizes]
      else:
        batch_sizes = batch_size * self.params["batch_multiplier"]


      def key_func(dict_):
        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        src_len = dict_["source_len"]
        tgt_len = dict_["target_len"]

        seq_len = tf.maximum(src_len, tgt_len)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_len),
            tf.less(seq_len, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))

        return bucket_id

      def reduce_func(bucket_id, windowed_data):
        return batching_func(batch_size_func(bucket_id), windowed_data)

      def window_size_func(bucket_id):
        return batch_size_func(bucket_id)
      
      # batch_size for different bucket_id
      # Used when batch_by_num_words is enabled
      # In this case, the given batch_size represents the num of words in each batch
      #
      #bucket_boundaries = [max(1, i * bucket_width) for i in range(num_buckets)]
      def batch_size_func(bucket_id):
        if isinstance(batch_sizes, list):
          batch_sizes_tensor = tf.constant(batch_sizes, dtype=tf.int64)
          return batch_sizes_tensor[bucket_id]
        return batch_sizes

      batched_dataset = src_tgt_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func, reduce_func=reduce_func, 
              window_size=None, window_size_func=window_size_func)
      )
    else:
      batched_dataset = batching_func(
          batch_size * self.params["batch_multiplier"], 
          src_tgt_dataset)
    

    def _pad_to_eight(tensor, pad_more=0):
      axis=1
      pad_value = eos
      
      shape = common_utils.shape_list(tensor)
      max_len = shape[axis]
      extra_len = tf.mod(8 - tf.mod(max_len, 8), 8)
      extra_len += pad_more

      paddings = [[0,0]] * len(shape)
      paddings[axis] = [0, extra_len]
      paddings = tf.convert_to_tensor(paddings)

      tensor = tf.pad(
          tensor, paddings,
          constant_values=pad_value)

      return tensor
    
    if self.params["pad_to_eight"]:
      batched_dataset = batched_dataset.map(
          lambda dict_: dict_.update({
            "source_tokens": _pad_to_eight(dict_["source_tokens"]),
            "target_tokens": _pad_to_eight(dict_["target_tokens"], pad_more=1),
          }) or dict_,
          num_parallel_calls=8).prefetch(1000)

    return batched_dataset
 

  def read_data(self, seed=None, **kwargs):
    src_files = self.params["source_files"]
    tgt_files = self.params["target_files"]
    
    # Dataset will be moved from contrib to tensorflow core
    # So we keep using it
    src_dataset = tf.data.TextLineDataset(src_files.split(","))
    tgt_dataset = tf.data.TextLineDataset(tgt_files.split(","))
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    batched_dataset = self._create_batched_dataset(
        src_tgt_dataset, seed=seed, **kwargs)

    # prefetch 1000 batches
    batched_dataset = batched_dataset.prefetch(1000)
    batched_iter = batched_dataset.make_one_shot_iterator()
    # return features
    features = batched_iter.get_next()
    return features
