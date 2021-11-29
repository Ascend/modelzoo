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

""" Generates model predictions.
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

import tensorflow as tf

from noahnmt.utils import flags as nmt_flags
from noahnmt.utils import cloud_utils
from noahnmt.utils import decode_utils
from noahnmt.utils import trainer_lib
from noahnmt.bin import infer

try:
  LOCAL_CACHE_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
  import moxing as mox
except KeyError:
  tf.logging.info("Local machine mode")


flags = tf.flags
FLAGS = flags.FLAGS

tf.flags.DEFINE_integer("min_step", 0, "min steps.")
tf.flags.DEFINE_integer("max_step", 0, "max steps. default 0 means inf")
tf.flags.DEFINE_integer("last_n", 0, "last n ckpts.")
tf.flags.DEFINE_integer("best_n", 0, "best n ckpts.")

def main(_argv):
  """Program entry point.
  """
  # set log_file
  if FLAGS.log_file:
    cloud_utils.set_log_file(FLAGS.log_file)

  # There might be several model_dirs in ensemble decoding
  model_dirs = FLAGS.model_dir.strip().split(",")
  assert len(model_dirs) == 1

  # update flags and params
  infer.load_config_and_update_flags()
  hparams = trainer_lib.create_hparams_from_flags(FLAGS)

  #
  model_name, model_params = decode_utils.update_model_name_and_params(
      hparams)

  # all checkpoint
  checkpoints = decode_utils.find_all_checkpoints(
      FLAGS.model_dir,
      min_step=FLAGS.min_step,
      max_step=FLAGS.max_step,
      last_n=FLAGS.last_n)

  # iterate over all checkpoints and translate
  global_step_tensor = 0
  best_bleu = 0
  best_step = 0
  best_n = []
  for counter, (step, ckpt) in enumerate(checkpoints):
    tf.logging.info("[%d] Translating using %s " % (counter, ckpt))

    # create model and translate using the given checkpoint
    hparams.checkpoint_path = ckpt
    decode_utils.create_model_and_translate(
        model_name, model_params, hparams)

    # calc BLEU
    bleu = decode_utils.calc_bleu(hparams)
    best_n.append((bleu, step, ckpt))
    tf.logging.info("BLEU at step %d: %.2f" % (step, bleu))
    if bleu > best_bleu:
      best_bleu = bleu
      best_step = step

    tf.reset_default_graph()

  tf.logging.info("Eval End!")
  tf.logging.info("Best BLEU %.2f at step %d" % (best_bleu, best_step))

  # save best-n ckpts
  if FLAGS.eval_keep_best_n:
    best_dir = os.path.join(FLAGS.model_dir, "best_ckpts")
    if tf.gfile.Exists(best_dir):
      tf.gfile.DeleteRecursively(best_dir)
    tf.gfile.MakeDirs(best_dir)

    # copy train_options
    tf.gfile.Copy(
        os.path.join(FLAGS.model_dir, "train_options.json"),
        os.path.join(best_dir, "train_options.json"),
        overwrite=True)

    FLAGS.eval_keep_best_n = min([FLAGS.eval_keep_best_n, len(best_n)])
    tf.logging.info("Saving best %d checkpoints to %s" % (FLAGS.eval_keep_best_n, best_dir))
    best_n = sorted(best_n, key=lambda x: x[0], reverse=True)[:FLAGS.eval_keep_best_n]
    for i, (bleu, step, ckpt) in enumerate(best_n):
      tf.logging.info("[%d] BLEU=%f model=%s" % (i, bleu, ckpt))
      try:
        filenames = tf.gfile.Glob(ckpt + ".*")
      except tf.errors.NotFoundError:
        filenames = tf.gfile.Glob(ckpt + ".*")
      for name in filenames:
        new_name = os.path.join(best_dir, name.split(os.sep)[-1])
        if name.startswith("s3://"):
            mox.file.copy(name, new_name)
        else:
            tf.gfile.Copy(name, new_name, overwrite=True)
    tf.logging.info("Done.")



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
