# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import heapq
import json
import math
import multiprocessing
import os
import signal
import typing

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import data_preprocessing
from official.recommendation import movielens
from official.recommendation import ncf_common
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
import time
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

from official.recommendation import logger_new
from official.recommendation import validator

FLAGS = flags.FLAGS

rank_size = int(os.getenv('RANK_SIZE'))

def construct_estimator(model_dir, params):
  """Construct either an Estimator or TPUEstimator for NCF.

  Args:
    model_dir: The model directory for the estimator
    params: The params dict for the estimator

  Returns:
    An Estimator or TPUEstimator.
  """
  distribution = ncf_common.get_v1_distribution_strategy(params)
  if params["accelerator"] == "gpu":
    run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                       eval_distribute=distribution)
  # for npu
  if params["accelerator"] == "1980":
  	session_config = tf.ConfigProto(
           inter_op_parallelism_threads=10,
           intra_op_parallelism_threads=10,
           allow_soft_placement=True)
  	npu_config = NPURunConfig(save_summary_steps=10000, enable_data_pre_proc=True, precision_mode="allow_mix_precision", save_checkpoints_steps=10000, session_config=session_config, model_dir=model_dir, iterations_per_loop=10, keep_checkpoint_max=5, hcom_parallel=True)
  model_fn = neumf_model.neumf_model_fn
  if params["use_xla_for_gpu"]:
    # TODO(seemuch): remove the contrib imput
    from tensorflow.contrib.compiler import xla
    logging.info("Using XLA for GPU for training and evaluation.")
    model_fn = xla.estimator_model_fn(model_fn)
  if params["accelerator"] == "gpu":
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       config=run_config, params=params)
  # for npu
  if params["accelerator"] == "1980":
    estimator=NPUEstimator(
      model_fn=model_fn,
      config=npu_config,
      model_dir=model_dir,
      params=params
  )

  return estimator


def log_and_get_hooks(eval_batch_size):
  """Convenience function for hook and logger creation."""
  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      FLAGS.hooks,
      model_dir=FLAGS.model_dir,
      batch_size=FLAGS.batch_size,  # for ExamplesPerSecondHook
      tensors_to_log={"cross_entropy": "cross_entropy"}
  )
  run_params = {
      "batch_size": FLAGS.batch_size,
      "eval_batch_size": eval_batch_size,
      "number_factors": FLAGS.num_factors,
      "hr_threshold": FLAGS.hr_threshold,
      "train_epochs": FLAGS.train_epochs,
  }
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="recommendation",
      dataset_name=FLAGS.dataset,
      run_params=run_params,
      test_id=FLAGS.benchmark_test_id)
  train_hooks.append(logger_new.LogSessionRunHook(FLAGS.print_freq))
  
  #loss_validation_hook = validator.LossValidationHook()
  #train_hooks.append(loss_validation_hook)

  return benchmark_logger, train_hooks


def main(_):
  with logger.benchmark_context(FLAGS), \
       mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    run_ncf(FLAGS)


def run_ncf(_):
  """Run NCF training and eval loop."""
  params = ncf_common.parse_flags(FLAGS)
  if rank_size > 1:
      params["batches_per_step"] = rank_size
  num_users, num_items, num_train_steps, num_eval_steps, producer = (
      ncf_common.get_inputs(params))
  if rank_size > 1:
  	  num_train_steps = num_train_steps/rank_size
  print("num_train_steps = %d" % num_train_steps)
  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  estimator = construct_estimator(model_dir=FLAGS.model_dir, params=params)

  benchmark_logger, train_hooks = log_and_get_hooks(params["eval_batch_size"])
  total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals

  target_reached = False
  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.TRAIN_LOOP)


  if params["debug_mod"]:
    train_input_fn = producer.make_input_fn(is_training=True)
    estimator.train(input_fn=train_input_fn, hooks=train_hooks,
                    steps=params["debug_steps"])   
  else: 
    for cycle_index in range(total_training_cycle):
        start_time = time.time()
        assert FLAGS.epochs_between_evals == 1 or not mlperf_helper.LOGGER.enabled
        logging.info("Starting a training cycle: {}/{}".format(
            cycle_index + 1, total_training_cycle))
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.TRAIN_EPOCH,
                                value=cycle_index)

        train_input_fn = producer.make_input_fn(is_training=True)
        estimator.train(input_fn=train_input_fn, hooks=train_hooks,
                        steps=num_train_steps)

        used_time = time.time() - start_time
        
        logging.info(
            "Epoch {}: Train_Used_time = {:.4f}".format(
                cycle_index, used_time)) 
        
        logging.info("Beginning evaluation.")
        eval_start_time = time.time()
        eval_input_fn = producer.make_input_fn(is_training=False)

        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_START,
                                value=cycle_index)
        eval_results = estimator.evaluate(eval_input_fn, steps=num_eval_steps)
        logging.info("Evaluation complete.")
        eval_used_time = time.time()-eval_start_time
        logging.info(
            "Epoch {}: Eval_Used_time = {:.4f}".format(
                cycle_index, eval_used_time)) 

        hr = float(eval_results[rconst.HR_KEY])
        ndcg = float(eval_results[rconst.NDCG_KEY])
        loss = float(eval_results["loss"])

        mlperf_helper.ncf_print(
            key=mlperf_helper.TAGS.EVAL_TARGET,
            value={"epoch": cycle_index, "value": FLAGS.hr_threshold})
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_ACCURACY,
                                value={"epoch": cycle_index, "value": hr})
        mlperf_helper.ncf_print(
            key=mlperf_helper.TAGS.EVAL_HP_NUM_NEG,
            value={"epoch": cycle_index, "value": rconst.NUM_EVAL_NEGATIVES})

        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_STOP, value=cycle_index)

        # Benchmark the evaluation results
        benchmark_logger.log_evaluation_result(eval_results)
        # Log the HR and NDCG results.
        logging.info(
            "Iteration {}: HR = {:.4f}, NDCG = {:.4f}, Loss = {:.4f}".format(
                cycle_index + 1, hr, ndcg, loss))
        # If some evaluation threshold is met
        if model_helpers.past_stop_threshold(FLAGS.hr_threshold, hr):
            target_reached = True
            break
  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.RUN_STOP,
                          value={"success": target_reached})
  producer.stop_loop()
  producer.join()

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()
  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.RUN_FINAL)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  ncf_common.define_ncf_flags()
  absl_app.run(main)
