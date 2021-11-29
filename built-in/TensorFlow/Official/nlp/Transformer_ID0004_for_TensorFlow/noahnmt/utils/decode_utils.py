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

import os
import time
import sys

import tensorflow as tf

from noahnmt.utils import data_utils
from noahnmt.utils import train_utils
from noahnmt.utils import trainer_lib
from noahnmt.configurable import _deep_merge_dict
from noahnmt.bin import train as nmt_train
from noahnmt.metrics import multi_bleu
from noahnmt.utils import registry
from noahnmt.utils import hook_utils

try:
  LOCAL_CACHE_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
  import moxing as mox
except KeyError:
  tf.logging.info("Local machine mode")

def checkpoint_exists(path):
  return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
          tf.gfile.Exists(path + ".index"))


def find_all_checkpoints(model_dir, min_step=0, max_step=0, last_n=0):
  """ find all checkpoints within [min_step, max_step]

  Return:
    list of tuple (step, path)
  """
  path_prefix = model_dir
  path_suffix = ".index"
  if not path_prefix.endswith(os.sep) and tf.gfile.IsDirectory(path_prefix):
    path_prefix += os.sep
  pattern = path_prefix + "model.ckpt-[0-9]*" + path_suffix

  try:
    checkpoints = tf.gfile.Glob(pattern)
  except tf.errors.NotFoundError:
    checkpoints = tf.gfile.Glob(pattern)

  if len(checkpoints) < 1:
    raise ValueError("Do not find checkpoints!")

  checkpoints = [name[:-len(path_suffix)] for name in checkpoints]
  checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
  # sort according to steps
  checkpoints = [(int(name.rsplit("-")[-1]), name) for name in checkpoints]
  checkpoints = [(step, name) for step, name in checkpoints if step >= min_step]
  if max_step > 0:
    checkpoints = [(step, name) for step, name in checkpoints if step <= max_step]
  if len(checkpoints) < 1:
    raise ValueError("Do not find checkpoints!")

  checkpoints = sorted(checkpoints, key=lambda x: x[0])
  if last_n > 0 and len(checkpoints) > last_n:
    checkpoints = checkpoints[-last_n:]

  return checkpoints



def calc_bleu(hparams):
  hyp_file = None
  for task in hparams.tasks:
    print("&&"*100)
    print(task)
    if task["class"] == "DecodeText":
      hyp_file = task["params"]["output"]
#   if not hyp_file or not tf.gfile.Exists(hyp_file):
#     tf.logging.warning("No output file found for BLEU calculation.")
#     sys.exit()
  if hyp_file.startswith("s3://"):
    tmp_file = os.path.join(LOCAL_CACHE_DIR, "output")
    mox.file.copy(hyp_file, tmp_file)
    hyp_file = tmp_file

  return multi_bleu.calc_bleu(
            os.path.abspath(hyp_file),
            os.path.abspath(hparams.reference),
            script=os.path.abspath(hparams.bleu_script))


def update_model_name_and_params(hparams, model_dirs=[]):
  # Load saved training options
  train_options = train_utils.TrainOptions.load(hparams.model_dir)
  model_name = train_options.model_name
  model_params = _deep_merge_dict(
       train_options.model_params, hparams.model_params)

  if len(model_dirs) > 1:
    model_name = "ensemble_seq2seq"
    model_params.update({
        "model_dirs": model_dirs})

  return model_name, model_params


def create_model_and_translate(model_name, model_params, hparams):
  # run_config and estimator
  run_config = nmt_train.create_run_config(hparams)
  estimator = trainer_lib.create_estimator(
      model_name=model_name,
      model_params=model_params,
      run_config=run_config,
      hparams=hparams)
  
  decode_hooks = hook_utils.create_hooks_from_dict(
      hooks_dict=hparams.hooks, 
      config=run_config, 
      hparams=hparams)

  # input fn
  mode = tf.estimator.ModeKeys.PREDICT
  input_fn = data_utils.make_estimator_input_fn(
      hparams.input_pipeline,
      mode, hparams)

  if hparams.score:
    mode = tf.estimator.ModeKeys.EVAL

  # Load inference tasks
  infer_tasks = []
  for tdict in hparams.tasks:
    if not "params" in tdict:
      tdict["params"] = {}
    task_cls = registry.class_ins(tdict["class"])
    task = task_cls(tdict["params"])
    infer_tasks.append(task)


  # translate
  predictions_iter = estimator.predict(
      input_fn,
      checkpoint_path=hparams.checkpoint_path,
      yield_single_examples=False,
      mode=mode,
      hooks=decode_hooks,
      init_from_scaffold=hparams.init_from_scaffold)

  # some help variables
  start_time = time.time()
  total_time_per_step = 0
  total_sents = 0
  total_steps = 0
  total_toks = 0

  # help func for recoding decoding time
  def gen_timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  # translate and postproc
  for elapsed_time, predictions in gen_timer(predictions_iter):
    for task in infer_tasks:
      task(predictions)

    total_time_per_step += elapsed_time
    total_steps += predictions["predicted_ids"].shape[-1]
    total_sents += predictions["predicted_ids"].shape[0]
    total_toks += predictions["predicted_ids"].shape[0] * predictions["predicted_ids"].shape[-1]

  # finalize
  for task in infer_tasks:
    task.finalize()

  tf.logging.info("Elapsed Time: %5.5f" % (time.time() - start_time))
  tf.logging.info("Generation Time per Sentence: %5.7f" %
                  (total_time_per_step / total_sents))
  tf.logging.info("Generation Time per Step: %5.7f" %
                  (total_time_per_step / total_steps))
  tf.logging.info("Generation Time perl Token: %5.7f" %
                  (total_time_per_step / total_toks))
