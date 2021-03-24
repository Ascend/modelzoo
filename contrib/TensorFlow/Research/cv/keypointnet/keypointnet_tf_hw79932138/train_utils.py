# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Utility functions for KeypointNet.

These are helper / tensorflow related functions. The actual implementation and
algorithm is in main.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
import tf_slim as slim
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator, NPUEstimatorSpec


class TrainingHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""
    def __init__(self, steps):
        self.steps = steps

        self.last_time = time.time()
        self.last_est = self.last_time

        self.eta_interval = int(math.ceil(0.1 * self.steps))
        self.current_interval = 0
        self.duration = 0

    def before_run(self, run_context):
        """Called before each call to run()."""
        graph = tf.get_default_graph()
        return tf.train.SessionRunArgs({
            "loss":
            graph.get_collection("total_loss")[0],
            "loss_con":
            graph.get_collection("loss_con")[0],
            "loss_angular":
            graph.get_collection("loss_angular")[0],
            "loss_sep":
            graph.get_collection("loss_sep")[0],
            "loss_sill":
            graph.get_collection("loss_sill")[0],
            "loss_variance":
            graph.get_collection("loss_variance")[0],
            "loss_lr":
            graph.get_collection("loss_lr")[0],
        })

    def after_run(self, run_context, run_values):
        """Called after each call to run()."""
        step = run_context.session.run(tf.train.get_global_step())
        now = time.time()

        if self.current_interval < self.eta_interval:
            self.duration = now - self.last_est
            self.current_interval += 1
        if step % self.eta_interval == 0:
            self.duration = now - self.last_est
            self.last_est = now

        eta_time = float(self.steps - step) / self.current_interval * \
            self.duration
        m, s = divmod(eta_time, 60)
        h, m = divmod(m, 60)
        eta = "%d:%02d:%02d" % (h, m, s)

        print("%.2f%% (%d/%d): loss:%.3e time:%.3f  end:%s (%s)" %
              (step * 100.0 / self.steps, step, self.steps,
               run_values.results["loss"], now - self.last_time,
               time.strftime("%a %d %H:%M:%S",
                             time.localtime(time.time() + eta_time)), eta))

        print(
            "loss_con:%.3e  loss_angular:%.3e  loss_sep:%.3e  loss_sill:%.3e  loss_variance:%.3e  loss_lr:%.3e"
            % (run_values.results["loss_con"],
               run_values.results["loss_angular"],
               run_values.results["loss_sep"], run_values.results["loss_sill"],
               run_values.results["loss_variance"],
               run_values.results["loss_lr"]))
        self.last_time = now


def standard_model_fn(func,
                      steps,
                      run_config=None,
                      sync_replicas=0,
                      optimizer_fn=None):
    """Creates model_fn for tf.Estimator.

    Args:
      func: A model_fn with prototype model_fn(features, labels, mode, hparams).
      steps: Training steps.
      run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
      sync_replicas: The number of replicas used to compute gradient for
          synchronous training.
      optimizer_fn: The type of the optimizer. Default to Adam.

    Returns:
      model_fn for tf.estimator.Estimator.
    """
    def fn(features, labels, mode, params):
        """Returns model_fn for tf.estimator.Estimator."""

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        ret = func(features, labels, mode, params)

        tf.add_to_collection("total_loss", ret["loss"])
        tf.add_to_collection("loss_con", ret["loss_con"])
        tf.add_to_collection("loss_angular", ret["loss_angular"])
        tf.add_to_collection("loss_sep", ret["loss_sep"])
        tf.add_to_collection("loss_sill", ret["loss_sill"])
        tf.add_to_collection("loss_variance", ret["loss_variance"])
        tf.add_to_collection("loss_lr", ret["loss_lr"])
        train_op = None

        training_hooks = []
        if is_training:
            training_hooks.append(TrainingHook(steps))

            if optimizer_fn is None:
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
            else:
                optimizer = optimizer_fn

            if run_config is not None and run_config.num_worker_replicas > 1:
                sr = sync_replicas
                if sr <= 0:
                    sr = run_config.num_worker_replicas

                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=sr,
                    total_num_replicas=run_config.num_worker_replicas)

                training_hooks.append(
                    optimizer.make_session_run_hook(
                        run_config.is_chief,
                        num_tokens=run_config.num_worker_replicas))

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer, 5)
            train_op = slim.learning.create_train_op(ret["loss"], optimizer)

        if "eval_metric_ops" not in ret:
            ret["eval_metric_ops"] = {}

        return NPUEstimatorSpec(mode=mode,
                                predictions=ret["predictions"],
                                loss=ret["loss"],
                                train_op=train_op,
                                eval_metric_ops=ret["eval_metric_ops"],
                                training_hooks=training_hooks)

    return fn


def train_and_eval(model_dir,
                   steps,
                   batch_size,
                   model_fn,
                   input_fn,
                   hparams,
                   keep_checkpoint_every_n_hours=0.5,
                   save_checkpoints_secs=180,
                   save_summary_steps=50,
                   sync_replicas=0):
    """Trains and evaluates our model. Supports local and distributed training.

    Args:
      model_dir: The output directory for trained parameters, checkpoints, etc.
      steps: Training steps.
      batch_size: Batch size.
      model_fn: A func with prototype model_fn(features, labels, mode, hparams).
      input_fn: A input function for the tf.estimator.Estimator.
      hparams: tf.HParams containing a set of hyperparameters.
      keep_checkpoint_every_n_hours: Number of hours between each checkpoint
          to be saved.
      save_checkpoints_secs: Save checkpoints every this many seconds.
      save_summary_steps: Save summaries every this many steps.
      eval_steps: Number of steps to evaluate model.
      eval_start_delay_secs: Start evaluating after waiting for this many seconds.
      eval_throttle_secs: Do not re-evaluate unless the last evaluation was
          started at least this many seconds ago
      sync_replicas: Number of synchronous replicas for distributed training.

    Returns:
      None
    """

    config = tf.ConfigProto(inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    run_config = NPURunConfig(
        model_dir=model_dir,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        session_config=config,
        save_checkpoints_secs=save_checkpoints_secs,
        save_summary_steps=save_summary_steps)

    estimator = NPUEstimator(model_dir=model_dir,
                             model_fn=standard_model_fn(
                                 model_fn,
                                 steps,
                                 run_config,
                                 sync_replicas=sync_replicas),
                             params=hparams,
                             config=run_config)

    estimator.train(input_fn=input_fn(split="train", batch_size=batch_size),
                    max_steps=steps)
