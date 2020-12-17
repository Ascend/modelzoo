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

""" translate a file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate
import sys
import os
import time
import numpy as np
import yaml

import tensorflow as tf

from noahnmt.bin import train as nmt_train
from noahnmt.configurable import _maybe_load_yaml, _deep_merge_dict
from noahnmt.utils import cloud_utils
from noahnmt.utils import device_utils
from noahnmt.utils import flags as nmt_flags
from noahnmt.utils import trainer_lib
from noahnmt.utils import decode_utils
# import moxing as mox

flags = tf.flags
FLAGS = flags.FLAGS

try:
  LOCAL_CACHE_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
  import moxing as mox
except KeyError:
  tf.logging.info("Local machine mode")


def load_config_and_update_flags():
  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.tasks = _maybe_load_yaml(FLAGS.tasks)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline = _maybe_load_yaml(FLAGS.input_pipeline)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)

  # reset gpu num
  FLAGS.worker_gpu = device_utils.get_num_gpus()
  tf.logging.info("")
  tf.logging.info("The number of worker GPUs: %d\n" % FLAGS.worker_gpu)

  # Load flags from config file
  final_config = {}
  if FLAGS.config_paths:
    for config_path in FLAGS.config_paths.split(","):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info("Loading config from %s", config_path)
      with tf.gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = _deep_merge_dict(final_config, config_flags)

  tf.logging.info("Final Config:\n%s", yaml.dump(final_config))

  # Merge flags with config values
  for flag_key, flag_value in final_config.items():
    if hasattr(FLAGS, flag_key) and isinstance(getattr(FLAGS, flag_key), dict):
      merged_value = _deep_merge_dict(flag_value, getattr(FLAGS, flag_key))
      setattr(FLAGS, flag_key, merged_value)
    elif hasattr(FLAGS, flag_key):
      setattr(FLAGS, flag_key, flag_value)
    else:
      tf.logging.warning("Ignoring config flag: %s", flag_key)


def main(_argv):
  """Program entry point.
  """
  # set log_file
  if FLAGS.log_file:
    cloud_utils.set_log_file(FLAGS.log_file)

  # if s3, copy to cache
  if FLAGS.reference and FLAGS.reference.startswith("s3://"):
    tf.logging.info("Copy references from S3 to local dir: %s" % LOCAL_CACHE_DIR)
    tmp_file = os.path.join(LOCAL_CACHE_DIR, "reference")

    if mox.file.exists(FLAGS.reference):
      mox.file.copy(FLAGS.reference, tmp_file)
    else:
      # multiple references
      MAX_REF_NUM = 20
      for i in range(MAX_REF_NUM):
        tmp_i = tmp_file + str(i)
        orig_i = FLAGS.reference + str(i)
        if mox.file.exists(orig_i):
          mox.file.copy(orig_i, tmp_i)
        else:
          break
    # set to local file
    FLAGS.reference = tmp_file

  # There might be several model_dirs in ensemble decoding
  model_dirs = FLAGS.model_dir.strip().split(",")
  FLAGS.model_dir = model_dirs[0]

  # Load flags from config file
  load_config_and_update_flags()
  hparams = trainer_lib.create_hparams_from_flags(FLAGS)
  hparams.init_from_scaffold = len(model_dirs)>1

  # update model name and model_params
  model_name, model_params = decode_utils.update_model_name_and_params(
      hparams, model_dirs=model_dirs)
  
  # 
  decode_utils.create_model_and_translate(
      model_name, model_params, hparams)

  if FLAGS.reference:
    bleu = decode_utils.calc_bleu(hparams)
    tf.logging.info("BLEU %.2f" % (bleu))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
