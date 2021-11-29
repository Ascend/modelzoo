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

import os

import sys
import tempfile
import json
import yaml
import contextlib

import tensorflow as tf
import moxing.tensorflow as mox
import moxing.tensorflow.executor.ops_adapter as ops_adapter


from noahnmt.utils import flags as nmt_flags  # pylint: disable=unused-import
from noahnmt.utils import registry
from noahnmt.configurable import _deep_merge_dict
from noahnmt.configurable import _maybe_load_yaml
from noahnmt.utils import data_utils
from noahnmt.utils import device_utils
from noahnmt.utils import trainer_lib
from noahnmt.utils import train_utils
from noahnmt.models import seq2seq_model
from noahnmt.utils import constant_utils
from noahnmt.layers import common_layers as common_utils
from noahnmt.utils import learning_rate_utils as lr_utils
from noahnmt.utils import graph_utils
from noahnmt.hooks import validation_hook
from noahnmt.hooks import obs_hook


flags = tf.flags
FLAGS = flags.FLAGS


def load_config_and_update_flags():
  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline_train = _maybe_load_yaml(FLAGS.input_pipeline_train)
  FLAGS.input_pipeline_dev = _maybe_load_yaml(FLAGS.input_pipeline_dev)

  # reset gpu num
  FLAGS.worker_gpu = 1 #device_utils.get_num_gpus()
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



def maybe_log_registry_and_exit():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)

    

def mox2tf_mode(run_mode):
  if run_mode == mox.ModeKeys.TRAIN:
    return tf.estimator.ModeKeys.TRAIN
  elif run_mode == mox.ModeKeys.EVAL:
    return tf.estimator.ModeKeys.EVAL
  elif run_mode == mox.ModeKeys.PREDICT:
    return tf.estimator.ModeKeys.PREDICT
  else:
    raise ValueError("Unknown mode " + str(run_mode))



# class MoxValidationHook(mox.AggregativeSessionRunHook):
#   def __init__(self, hook_params):
#     mox.AggregativeSessionRunHook.__init__(self)
#     self._valid_hook = 

#   def before_run(self, run_context):
#     feed_x = 1.0
#     return tf.train.SessionRunArgs(fetches=None, feed_dict={x: feed_x})

#   def support_aggregation(self):
#     return False

#   def support_sync_workers(self):
#     return False

#   def run_inter_mode(self):
#     return False


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  maybe_log_registry_and_exit()

  # load config file and params
  # and update flags so that flags stores complete and newest params
  load_config_and_update_flags()
  # convert FLAGS to hparams
  hparams = trainer_lib.create_hparams_from_flags(FLAGS)

  mox.file.make_dirs("/cache/local_log")
  tf.logging.set_verbosity(tf.logging.INFO)
  mox.set_flag('gradient_clip_type', 'norm')
  mox.set_flag('gradient_clip', hparams.gradient_clip)
  mox.set_flag('sync_before_server', True)
  mox.set_flag('session_timeout', 0)
  mox.file.set_auth(retry=50)
  mox.set_flag('num_inter_threads',0)
  mox.set_flag('num_intra_threads',0)

  ###############
  def input_fn(run_mode, **kwargs):
    def_dict = {}
    run_mode = mox2tf_mode(run_mode)
    if run_mode == tf.estimator.ModeKeys.TRAIN:
      def_dict = hparams.input_pipeline_train
      
      # split data
      if "params" not in def_dict:
        def_dict["params"] = {}

      def_dict["params"]["num_shards"] = ops_adapter.size()
      def_dict["params"]["shard_index"] = ops_adapter.rank()

    elif run_mode == tf.estimator.ModeKeys.EVAL:
      def_dict = hparams.input_pipeline_dev
    else:
      raise ValueError("Unknown mode " + str(run_mode))

    _input_fn = data_utils.make_estimator_input_fn(
      def_dict=def_dict,
      mode=run_mode,
      hparams=hparams)
    
    # get Dataset iter and features
    features, _ = _input_fn(None, None)

    return features
  

  ###################
  def model_fn(inputs, run_mode, **kwargs):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    run_mode = mox2tf_mode(run_mode)

    # train_options used for creating models
    train_options = train_utils.TrainOptions(
        model_name=hparams.model_name,
        model_params=hparams.model_params)
    # On the main worker, save training options
    if ops_adapter.rank == 0:
      if not tf.gfile.Exists(hparams.model_dir):
        tf.gfile.MakeDirs(hparams.model_dir)
      train_options.dump(hparams.model_dir)
    
    #####
    mode_ = run_mode
    if hparams.eval_run_autoregressive and run_mode == tf.estimator.ModeKeys.EVAL:
      mode_ = tf.estimator.ModeKeys.PREDICT

    model_cls = registry.model(train_options.model_name)

    model = model_cls(
        train_options.model_params,
        mode_,
        data_parallelism=None,
        hparams=hparams)

    # TRAIN and EVAL modes
    decoder_output, losses_dict = model(inputs)

    # predictions
    predictions = model._create_predictions(
        decoder_output=decoder_output,
        features=inputs)
    graph_utils.add_dict_to_collection(predictions, "predictions")

    # log trainable variables
    # if mode == tf.estimator.ModeKeys.TRAIN:
    common_utils.print_trainable_variables(run_mode)

    loss = 0.
    if losses_dict:
      # Accumulate losses
      loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))
      tf.summary.scalar("loss", loss)
    else:
      loss = tf.constant(loss, dtype=constant_utils.DT_FLOAT())

    log_info = {'loss': loss}

    # add hooks
    hooks = []
    if hparams.schedule == "train_and_evaluate":
      # validation hook
      pass

    if hparams.model_dir.startswith("s3://"):
      hooks.append(
          obs_hook.CheckpointSyncToOBSHook(
            hparams.model_dir,
            save_secs=hparams.save_checkpoints_secs,
          )
      )

    return mox.ModelSpec(loss=loss, log_info=log_info, hooks=hooks)


  #############################
  def optimizer_fn():
    
    # get learning rate
    assert "embedding.dim" in hparams.model_params
    assert "learning_rate.schedule" in hparams.model_params
    assert "learning_rate.warmup_steps" in hparams.model_params
    assert "learning_rate.constant" in hparams.model_params

    params = dict(**hparams.model_params)
    params["learning_rate.start_decay_step"] = 0
    params["learning_rate.stop_decay_at"] = 0
    if "optimizer.params" not in params:
      params["optimizer.params"] = {}
    params["optimizer.params"]["accumulate_steps"] = 1

    learning_rate = lr_utils.learning_rate_schedule(params, hparams, 1)
        
    opt = mox.get_optimizer_fn("adam", learning_rate=learning_rate)()
    
    if hparams.use_fp16:
      assert "mixed_precision.params" in params
      mixed_precision_params = params["mixed_precision.params"]

      loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
          init_loss_scale=mixed_precision_params["init_loss_scale"],
          incr_every_n_steps=mixed_precision_params["incr_every_n_steps"],
          decr_every_n_nan_or_inf=mixed_precision_params["decr_every_n_nan_or_inf"],
          incr_ratio=mixed_precision_params["incr_ratio"],
          decr_ratio=mixed_precision_params["decr_ratio"])

      opt_tmp = opt
      opt = tf.contrib.mixed_precision.LossScaleOptimizer(opt_tmp, loss_scale_manager)
      opt._lr = opt_tmp._lr
      opt._lr_t = opt_tmp._lr_t

    return opt

  # local model dir
  # model_dir = "/cache/model_dir"
  model_dir = hparams.model_dir
  if hparams.efs_model_dir:
    model_dir = hparams.efs_model_dir
    if hparams.clear_efs:
      mox.file.rmove(model_dir, True)

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=optimizer_fn,
          run_mode=mox.ModeKeys.TRAIN,
          log_dir=model_dir,
          max_number_of_steps=hparams.train_steps,
          log_every_n_steps=100,
          save_model_secs=hparams.save_checkpoints_secs,
          save_summary_steps=100,
          auto_batch=False)
  
  if hparams.efs_model_dir and ops_adapter.rank() == 0:
      mox.file.copy_parallel(hparams.efs_model_dir, hparams.model_dir)
  # mox.file.copy_parallel(model_dir, hparams.model_dir)



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
