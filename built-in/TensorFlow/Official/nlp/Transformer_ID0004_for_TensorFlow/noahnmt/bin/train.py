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

"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import os
import sys
import tempfile
import json
import yaml

import tensorflow as tf

from noahnmt.utils import flags as nmt_flags  # pylint: disable=unused-import
from noahnmt.utils import registry
from noahnmt.utils import trainer_lib
from noahnmt.utils import train_utils
from noahnmt.utils import device_utils
from noahnmt.configurable import _deep_merge_dict
from noahnmt.configurable import _maybe_load_yaml


flags = tf.flags
FLAGS = flags.FLAGS


# rank_size = os.environ.get('RANK_SIZE', '').strip()
# if int(rank_size) > 1:


def load_config_and_update_flags():
  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline_train = _maybe_load_yaml(FLAGS.input_pipeline_train)
  FLAGS.input_pipeline_dev = _maybe_load_yaml(FLAGS.input_pipeline_dev)

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
    elif hasattr(FLAGS, flag_key) and not getattr(FLAGS, flag_key):
      setattr(FLAGS, flag_key, flag_value)
    else:
      tf.logging.warning("Ignoring config flag: %s", flag_key)


def create_experiment_fn(**kwargs):
  return trainer_lib.create_experiment_fn(
      model_name=FLAGS.model_name,
      model_params=FLAGS.model_params,
      train_steps=FLAGS.train_steps,
      eval_steps=None,
      min_eval_frequency=FLAGS.eval_every_n_steps,
      schedule=FLAGS.schedule,
      eval_throttle_seconds=FLAGS.eval_throttle_seconds,
      export=False,
      use_tfdbg=FLAGS.tfdbg,
      use_dbgprofile=FLAGS.dbgprofile,
      eval_keep_best_n=FLAGS.eval_keep_best_n,
      eval_keep_best_metric=FLAGS.eval_keep_best_metric,
      eval_keep_best_metric_minimize=FLAGS.eval_keep_best_metric_minimize,
      eval_early_stopping_steps=FLAGS.early_stopping_rounds,
      eval_early_stopping_metric=FLAGS.early_stopping_metric,
      eval_early_stopping_metric_delta=FLAGS.early_stopping_metric_delta,
      eval_early_stopping_metric_minimize=FLAGS.early_stopping_metric_minimize,
      use_xla=FLAGS.xla_compile,
      warm_start_from=FLAGS.warm_start_from,
      **kwargs)


def create_run_config(hparams):
  """Create a run config.

  Args:
    hp: model hyperparameters
  Returns:
    a run config
  """
  save_ckpt_steps = FLAGS.eval_every_n_steps
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert FLAGS.model_dir or FLAGS.checkpoint_path

  # the various custom getters we have written do not play well together yet.
  # TODO(noam): ask rsepassi for help here.
  daisy_chain_variables = (
      FLAGS.daisy_chain_variables and not FLAGS.use_fp16)
  return trainer_lib.create_run_config(
      model_dir=os.path.expanduser(FLAGS.model_dir),
      master=FLAGS.master,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=save_ckpt_steps,
      save_checkpoints_secs=save_ckpt_secs,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      shard_to_cpu=FLAGS.locally_shard_to_cpu,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.enable_graph_rewriter,
      schedule=FLAGS.schedule,
      no_data_parallelism=(not FLAGS.data_parallelism),
      dp_param_shard=FLAGS.dp_param_shard,
      daisy_chain_variables=daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      inter_op_parallelism_threads=FLAGS.inter_op_threads,
      log_step_count_steps=FLAGS.log_step_count_steps,
      intra_op_parallelism_threads=FLAGS.intra_op_threads,
      over_dump=FLAGS.over_dump,
      over_dump_path=FLAGS.over_dump_path)


@contextlib.contextmanager
def profile_context():
  if FLAGS.profile:
    with tf.contrib.tfprof.ProfileContext(
        "nmtprof", trace_steps=range(20, 25), dump_steps=range(20,25)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(20,25))
      yield
  else:
    yield


def maybe_log_registry_and_exit():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def is_chief():
  schedules = ["train", "train_and_evaluate", "continuous_train_and_eval"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  with profile_context():
    getattr(exp, FLAGS.schedule)()


def run_std_server():
  exp = trainer_lib.NMTExperiment(*([None] * 5))
  exp.run_std_server()


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.schedule == "run_std_server":
    run_std_server()
  trainer_lib.set_random_seed(FLAGS.random_seed)
  maybe_log_registry_and_exit()

  # load config file and params
  # and update flags so that flags stores complete and newest params
  load_config_and_update_flags()

  # convert FLAGS to hparams
  hparams = trainer_lib.create_hparams_from_flags(FLAGS)

  # experiments
  exp_fn = create_experiment_fn()
  exp = exp_fn(create_run_config(hparams), hparams)
  execute_schedule(exp)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
