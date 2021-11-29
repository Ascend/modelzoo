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

""" TODO
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os

import six
import tensorflow as tf
import yaml
from tensorflow import gfile
from tensorflow.python.client import timeline  # pylint: disable=E0611
from tensorflow.python.profiler import option_builder


def _micro_anaylsis_options(output_dir, view):
  """Options for microsecond analysis
  """
  options = option_builder.ProfileOptionBuilder.trainable_variables_parameter()
  options["select"] = ["micros", "occurrence", "device"]
  options["min_micros"] = 1000
  options["account_type_regexes"] = [".*"]
  options["order_by"] = "micros"
  if output_dir:
    out_path = os.path.join(output_dir, "micro-%s.txt" % view)
    options["output"] = "file:outfile=%s" % out_path
  return options


def _profile_view(profiler, options, view):
  if view == "graph":
    profiler.profile_graph(options)
  elif view == "scope":
    profiler.profile_name_scope(options)
  elif view == "op":
    profiler.profile_operations(options)
  elif view == "code":
    profiler.profile_python(options)
  else:
    raise ValueError("Unknown profiling view: %s" % view)



def write_profiler(profiler, views, model_dir):
  output_dir = os.path.join(model_dir, "profile")
  tf.logging.info("Write profiling data to dir: %s" % output_dir)
  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)
  for view in views:
    profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
      options=_micro_anaylsis_options(output_dir, view))
    options = profile_graph_opts_builder.build()
    _profile_view(profiler, options, view)


def write_metadata(run_metadata, model_dir, step_done):
  output_dir = os.path.join(model_dir, "profile")
  tf.logging.info("Captured full trace at step %s", step_done)
  # Create output directory
  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

  # Save run metadata
  trace_path = os.path.join(output_dir, "run_meta-%d" % step_done)
  with gfile.GFile(trace_path, "wb") as trace_file:
    trace_file.write(run_metadata.SerializeToString())
    tf.logging.info("Saved run_metadata to %s", trace_path)

  # Save timeline
  timeline_path = os.path.join(output_dir, "timeline-%d.json" % step_done)
  with gfile.GFile(timeline_path, "w") as timeline_file:
    tl_info = timeline.Timeline(run_metadata.step_stats)
    tl_chrome = tl_info.generate_chrome_trace_format()
    timeline_file.write(tl_chrome)
    tf.logging.info("Saved timeline to %s", timeline_path)

  # Save tfprof op log
  # tf.profiler.write_op_log(
  #     graph=tf.get_default_graph(),
  #     log_dir=output_dir,
  #     run_meta=run_metadata)
  # tf.logging.info("Saved op log to %s", output_dir)

