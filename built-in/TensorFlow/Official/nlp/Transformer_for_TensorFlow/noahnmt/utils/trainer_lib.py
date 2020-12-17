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

"""Library for training. See train.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug

from noahnmt.utils import registry
from noahnmt.utils import data_utils
from noahnmt.utils import device_utils
from noahnmt.utils import parallel_utils
from noahnmt.utils import train_utils
from noahnmt.utils import hook_utils
from noahnmt.models import seq2seq_model
from noahnmt.layers import nmt_estimator
from noahnmt.hooks import validation_hook
from noahnmt.hooks import metrics_hook
# from noahnmt.hooks import multikd_validation_hook

# from tensorflow.contrib.offline_train.python.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import NPURunConfig


RANK_SIZE = int(os.environ.get('RANK_SIZE', '1').strip())
RANK_ID = int(os.environ.get('DEVICE_ID', '0').strip())


def next_checkpoint(model_dir, timeout_mins=120):
  """Yields successive checkpoints from model_dir."""
  last_ckpt = None
  while True:
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=60, timeout=60 * timeout_mins)

    if last_ckpt is None:
      tf.logging.info(
          "Eval timeout: no new checkpoints within %dm" % timeout_mins)
      break

    yield last_ckpt


def create_session_config(log_device_placement=False,
                          enable_graph_rewriter=False,
                          gpu_mem_fraction=0.95,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0):
  """The TensorFlow Session config to use."""
  if enable_graph_rewriter:
    rewrite_options = rewriter_config_pb2.RewriterConfig()
    rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
    graph_options = tf.GraphOptions(rewrite_options=rewrite_options)
  else:
    graph_options = tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
  
  custom_op=graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name =  "NpuOptimizer"
  custom_op.parameter_map["min_group_size"].b = 1
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["enable_data_pre_proc"].b = True
  custom_op.parameter_map["enable_auto_mix_precision"].b=False
  
  if RANK_SIZE > 1:
    custom_op.parameter_map["hcom_parallel"].b = True

  graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

  config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      gpu_options=gpu_options,
      log_device_placement=log_device_placement,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads)
  return config


def is_cloud_async_distributed():
  return ("chief" in
          json.loads(os.environ.get("TF_CONFIG", "{}")).get("cluster", {}))


def create_run_config(master="",
                      model_dir=None,
                      num_shards=8,
                      log_device_placement=False,
                      save_checkpoints_steps=1000,
                      save_checkpoints_secs=None,
                      keep_checkpoint_max=20,
                      keep_checkpoint_every_n_hours=10000,
                      num_gpus=1,
                      gpu_order="",
                      shard_to_cpu=False,
                      num_async_replicas=1,
                      enable_graph_rewriter=False,
                      gpu_mem_fraction=0.95,
                      no_data_parallelism=False,
                      dp_param_shard=True,
                      daisy_chain_variables=True,
                      schedule="continuous_train_and_eval",
                      worker_job="/job:localhost",
                      worker_id=0,
                      ps_replicas=0,
                      ps_job="/job:ps",
                      ps_gpu=0,
                      random_seed=None,
                      sync=False,
                      inter_op_parallelism_threads=0,
                      log_step_count_steps=100,
                      intra_op_parallelism_threads=0):
  """Create RunConfig, and Parallelism object."""
  
  """session_config = create_session_config(
      log_device_placement=log_device_placement,
      enable_graph_rewriter=enable_graph_rewriter,
      gpu_mem_fraction=gpu_mem_fraction,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads)"""
  

  session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["enable_data_pre_proc"].b = False
  custom_op.parameter_map["mix_compile_mode"].b = False
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["min_group_size"].b = 1
  custom_op.parameter_map["enable_auto_mix_precision"].b=False
  
  if RANK_SIZE > 1:
    custom_op.parameter_map["hcom_parallel"].b = True

  run_config_args = {
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 1,
      "save_checkpoints_steps": save_checkpoints_steps,
      "save_checkpoints_secs": save_checkpoints_secs,
      "keep_checkpoint_max": keep_checkpoint_max,
      "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
      "tf_random_seed": random_seed,
      "log_step_count_steps": 1
  }
  if save_checkpoints_secs:
    del run_config_args["save_checkpoints_steps"]
  # run_config_cls = tf.contrib.learn.RunConfig
  run_config_cls = NPURunConfig
  # run_config_args.update({
  #     "enable_data_pre_proc": False,
  # })

  config = run_config_cls(**run_config_args)

  # add device info for data_parallelism
  config.nmt_device_info = {
      "num_async_replicas": num_async_replicas,
  }
  config.data_parallelism = None

  return config


def create_estimator(model_name,
                     model_params,
                     run_config,
                     hparams,
                     schedule="train_and_evaluate",
                     use_xla=False):
  """Create a T2T Estimator."""
  model_fn = seq2seq_model.Seq2seqModel.make_estimator_model_fn(
      model_name, model_params, hparams)

  del use_xla

  estimator = nmt_estimator.NMTEstimator(
      model_fn=model_fn,
      model_dir=run_config.model_dir,
      config=run_config,
  )
  return estimator


def create_hooks(use_tfdbg=False,
                 use_dbgprofile=False,
                 dbgprofile_kwargs=None,
                 use_validation_monitor=False,
                 validation_monitor_kwargs=None,
                 use_early_stopping=False,
                 early_stopping_kwargs=None,
                 multikd_mode=False):
  """Create train and eval hooks for Experiment."""
  train_hooks = []
  eval_hooks = []

  if use_tfdbg:
    hook = debug.LocalCLIDebugHook()
    train_hooks.append(hook)
    eval_hooks.append(hook)

  if use_dbgprofile:
    # Recorded traces can be visualized with chrome://tracing/
    # The memory/tensor lifetime is also profiled
    tf.logging.info("Using ProfilerHook")
    defaults = dict(save_steps=10, show_dataflow=True, show_memory=True)
    defaults.update(dbgprofile_kwargs)
    train_hooks.append(tf.train.ProfilerHook(**defaults))

  if use_validation_monitor:
    if multikd_mode:
      tf.logging.info("Using MultikdValidationMonitor")
      # train_hooks.append(
      #     multikd_validation_hook.MultikdValidationHook(
      #         hooks=eval_hooks, **validation_monitor_kwargs))
    else:
      tf.logging.info("Using ValidationMonitor")
      train_hooks.append(
          validation_hook.ValidationHook(
              hooks=eval_hooks, **validation_monitor_kwargs))

  if use_early_stopping:
    tf.logging.info("Using EarlyStoppingHook")
    hook = metrics_hook.EarlyStoppingHook(**early_stopping_kwargs)
    # Adding to both training and eval so that eval aborts as well
    train_hooks.append(hook)
    eval_hooks.append(hook)

  return train_hooks, eval_hooks


class NMTExperiment(object):
  """Custom Experiment class for running distributed experiments."""

  def __init__(self, estimator, hparams, train_spec, eval_spec,
               use_validation_monitor):
    self._train_spec = train_spec
    self._eval_spec = eval_spec
    self._hparams = hparams
    self._estimator = estimator
    self._use_validation_monitor = use_validation_monitor

  @property
  def estimator(self):
    return self._estimator

  @property
  def train_steps(self):
    return self._train_spec.max_steps

  @property
  def eval_steps(self):
    return self._eval_spec.steps

  def continuous_train_and_eval(self, continuous_eval_predicate_fn=None):
    del continuous_eval_predicate_fn
    tf.estimator.train_and_evaluate(self._estimator, self._train_spec,
                                    self._eval_spec)
    return self.evaluate()

  def train_and_evaluate(self):
    if self._use_validation_monitor:
      tf.logging.warning("EvalSpec not provided. Estimator will not manage "
                         "model evaluation. Assuming ValidationMonitor present "
                         "in train_hooks.")
      self.train()
    # tf.estimator.train_and_evaluate(self._estimator, self._train_spec,
    #                                 self._eval_spec)
    # return self.evaluate()


  def train(self):
    self._estimator.train(
        self._train_spec.input_fn,
        hooks=self._train_spec.hooks,
        max_steps=self._train_spec.max_steps)

  def evaluate(self):
    return self._estimator.evaluate(
        self._eval_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks)

  def evaluate_on_train_data(self):
    self._estimator.evaluate(
        self._train_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks,
        name="eval_train")

  def continuous_eval(self):
    """Evaluate until checkpoints stop being produced."""
    for _ in next_checkpoint(self._hparams.model_dir):
      self.evaluate()

  def continuous_eval_on_train_data(self):
    """Evaluate on train data until checkpoints stop being produced."""
    for _ in next_checkpoint(self._hparams.model_dir):
      self.evaluate_on_train_data()

  def test(self):
    """Perform 1 step of train and 2 step of eval."""
    if self._use_validation_monitor:
      return self.train_and_evaluate()

    self._estimator.train(
        self._train_spec.input_fn, hooks=self._train_spec.hooks, max_steps=1)

    self._estimator.evaluate(
        self._eval_spec.input_fn, steps=1, hooks=self._eval_spec.hooks)

  def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    config = tf.estimator.RunConfig()
    server = tf.train.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id)
    server.join()

  # def decode(self, dataset_split=None, decode_from_file=False):
  #   """Decodes from dataset or file."""
  #   if decode_from_file:
  #     decoding.decode_from_file(self._estimator,
  #                               self._decode_hparams.decode_from_file,
  #                               self._hparams,
  #                               self._decode_hparams,
  #                               self._decode_hparams.decode_to_file)
  #   else:
  #     decoding.decode_from_dataset(self._estimator,
  #                                  self._hparams.problem.name,
  #                                  self._hparams,
  #                                  self._decode_hparams,
  #                                  dataset_split=dataset_split)

  # def continuous_decode(self):
  #   """Decode from dataset on new checkpoint."""
  #   for _ in next_checkpoint(self._hparams.model_dir):
  #     self.decode()

  # def continuous_decode_on_train_data(self):
  #   """Decode from dataset on new checkpoint."""
  #   for _ in next_checkpoint(self._hparams.model_dir):
  #     self.decode(dataset_split=tf.estimator.ModeKeys.TRAIN)

  # def continuous_decode_from_file(self):
  #   """Decode from file on new checkpoint."""
  #   for _ in next_checkpoint(self._hparams.model_dir):
  #     self.decode(decode_from_file=True)


def create_experiment(
    run_config,
    hparams,
    model_name,
    model_params,
    train_steps,
    eval_steps,
    min_eval_frequency=2000,
    eval_throttle_seconds=600,
    schedule="train_and_evaluate",
    export=False,
    use_tfdbg=False,
    use_dbgprofile=False,
    eval_keep_best_n=0,
    eval_keep_best_metric=None,
    eval_keep_best_metric_minimize=None,
    eval_early_stopping_steps=None,
    eval_early_stopping_metric=None,
    eval_early_stopping_metric_delta=None,
    eval_early_stopping_metric_minimize=True,
    use_xla=False,
    warm_start_from=None,
    decode_from_file=None,
    decode_to_file=None,
    decode_reference=None):
  """Create Experiment."""
  # train_options used for creating models
  train_options = train_utils.TrainOptions(
      model_name=model_name,
      model_params=model_params)
  # On the main worker, save training options
  if run_config.is_chief:
    if not tf.gfile.Exists(run_config.model_dir):
      tf.gfile.MakeDirs(run_config.model_dir)
    train_options.dump(run_config.model_dir)

  # Estimator
  estimator = create_estimator(
      train_options.model_name,
      train_options.model_params,
      run_config,
      hparams,
      schedule=schedule,
      use_xla=use_xla)

  # Input fns
  def_dict_train = hparams.input_pipeline_train
  # split data
  if "params" not in def_dict_train:
    def_dict_train["params"] = {}
  def_dict_train["params"]["num_shards"] = RANK_SIZE
  def_dict_train["params"]["shard_index"] = RANK_ID

  train_input_fn = data_utils.make_estimator_input_fn(
      def_dict=def_dict_train,
      mode=tf.estimator.ModeKeys.TRAIN,
      hparams=hparams)
  eval_input_fn = data_utils.make_estimator_input_fn(
      def_dict=hparams.input_pipeline_dev,
      mode=tf.estimator.ModeKeys.EVAL,
      hparams=hparams)
  
  # Export
  exporter = None
  if export:
    def compare_fn(best_eval_result, current_eval_result):
      metric = eval_early_stopping_metric or "loss"
      return current_eval_result[metric] < best_eval_result[metric]

    exporter = tf.estimator.BestExporter(
        name="best",
        serving_input_receiver_fn=lambda: data_utils.serving_input_fn(),
        compare_fn=compare_fn,
        assets_extra=None)

  # Hooks
  validation_monitor_kwargs = dict(
      estimator=estimator,
      input_fn=eval_input_fn,
      eval_steps=eval_steps,
      every_n_steps=min_eval_frequency,
      eval_keep_best_n=eval_keep_best_n,
      eval_keep_best_metric=eval_keep_best_metric,
      eval_keep_best_metric_minimize=eval_keep_best_metric_minimize,
      early_stopping_rounds=eval_early_stopping_steps,
      early_stopping_metric=eval_early_stopping_metric,
      early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
  dbgprofile_kwargs = {"output_dir": run_config.model_dir}
  early_stopping_kwargs = dict(
      events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
      tag=eval_early_stopping_metric,
      num_plateau_steps=eval_early_stopping_steps,
      plateau_decrease=eval_early_stopping_metric_minimize,
      plateau_delta=eval_early_stopping_metric_delta,
      every_n_steps=min_eval_frequency)

  # In-process eval (and possible early stopping)
  if schedule == "continuous_train_and_eval" and min_eval_frequency:
    tf.logging.warn("ValidationMonitor only works with "
                    "--schedule=train_and_evaluate")
  use_validation_monitor = (
      schedule == "train_and_evaluate" and min_eval_frequency)
  # Distributed early stopping
  local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
  # use_early_stopping = (
  #     schedule not in local_schedules and eval_early_stopping_steps)
  use_early_stopping = False
  
  multikd_mode = bool(hparams.multikd_lang_pairs)
  if multikd_mode:
    validation_monitor_kwargs.update(dict(
        lang_pairs=hparams.multikd_lang_pairs,
        teacher_scores=hparams.multikd_teacher_scores,
        multikd_delta=hparams.multikd_delta))
    if not use_validation_monitor:
      raise ValueError("In Multikd mode, should enable train_and_evaluate and validation_monitor")

  train_hooks, eval_hooks = create_hooks(
      use_tfdbg=use_tfdbg,
      use_dbgprofile=use_dbgprofile,
      dbgprofile_kwargs=dbgprofile_kwargs,
      use_validation_monitor=use_validation_monitor,
      validation_monitor_kwargs=validation_monitor_kwargs,
      use_early_stopping=use_early_stopping,
      early_stopping_kwargs=early_stopping_kwargs,
      multikd_mode=multikd_mode)
  
  train_hooks += hook_utils.create_hooks_from_dict(
      hooks_dict=hparams.hooks, 
      config=run_config, 
      hparams=hparams)
  
  train_hooks += hook_utils.create_helpful_hooks(run_config)

  
  train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      train_hooks, estimator)
  eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      eval_hooks, estimator)
  
  # move validation_hook to the end
  # valid_hook_idx = None
  # for i in range(len(train_hooks)):
  #   if isinstance(train_hooks[i], validation_hook.ValidationHook):
  #     valid_hook_idx = i
  #     break
  # if valid_hook_idx is not None:
  #   train_hooks.append(train_hooks.pop(valid_hook_idx))

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      throttle_secs=eval_throttle_seconds,
      exporters=exporter)

  return NMTExperiment(estimator, hparams, train_spec, eval_spec,
                       use_validation_monitor)


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def restore_checkpoint(ckpt_dir, saver, sess, must_restore=False):
  """Restore from a checkpoint."""
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if must_restore and not ckpt:
    raise ValueError("No checkpoint found in %s" % ckpt_dir)
  if not ckpt:
    return 0

  path = ckpt.model_checkpoint_path
  tf.logging.info("Restoring checkpoint %s", path)
  saver.restore(sess, path)
  step = int(path.split("-")[-1])
  return step


def create_hparams_from_flags(flags_):
  """ FLAGS to HParams for convinence
  """
  hparams = tf.contrib.training.HParams()
  flags_dict = flags_.__flags
  for key, f in flags_dict.items():
    if isinstance(f.value, (list, tuple)) and not f.value:
      hparams.add_hparam(key, None)
    else:
      hparams.add_hparam(key, f.value)
  return hparams
