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

"""Common command-line flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_bool(
    "registry_help", False, 
        "If True, logs the contents of the registry and exits.")
flags.DEFINE_bool(
    "tfdbg", False, 
        "If True, use the TF debugger CLI on train/eval.")
flags.DEFINE_bool(
    "dbgprofile", False, 
        "If True, record the timeline for chrome://tracing/.")
flags.DEFINE_bool(
    "profile", False, 
        "Profile performance?")
flags.DEFINE_integer(
    "random_seed", None, 
        "Random seed for TensorFlow initializers. Setting "
        "this value allows consistency between reruns.")

###### S3 on Huawei Cloud 
flags.DEFINE_string(
    "log_file", None, 
        "the file to store logs on s3. Error info will not be stored")

###### some common flags
flags.DEFINE_string(
    "model_dir", None, 
        "The directory to write model checkpoints and summaries to. "
        "If None, a local temporary directory is created.")

flags.DEFINE_boolean(
    "use_fp16", False, 
        "Use float16 variables and ops.")

###### Model Flags
flags.DEFINE_string(
    "config_paths", "",  
        "Path to a YAML configuration files defining FLAG "
        "values. Multiple files can be separated by commas. "
        "Files are merged recursively. Setting a key in these "
        "files is equivalent to setting the FLAG value with "
        "the same name.")
flags.DEFINE_string(
    "hooks", "[]", 
        "YAML configuration string for the training hooks to use.")
flags.DEFINE_string(
    "metrics", "[]", 
        "YAML configuration string for the training metrics to use.")
flags.DEFINE_string(
    "model_name", "",  
        "Name of the model class. "
        "Can be either a fully-qualified name, or the name "
        "of a class defined in `seq2seq.models`.")
flags.DEFINE_string(
    "model_params", "{}", 
        "YAML configuration string for the model parameters.")


###### session config
flags.DEFINE_bool(
    "xla_compile", False, 
        "Whether to use XLA to compile graph.")
flags.DEFINE_integer(
    "inter_op_threads", 0, 
        "Number of inter_op_parallelism_threads to use for CPU. "
        "See TensorFlow config.proto for details.")
flags.DEFINE_integer(
    "intra_op_threads", 0, 
        "Number of intra_op_parallelism_threads to use for CPU. "
        "See TensorFlow config.proto for details.")
flags.DEFINE_bool(
    "log_device_placement", False, 
        "Whether to log device placement.")
flags.DEFINE_bool(
    "enable_graph_rewriter", False, 
        "Enable graph optimizations that are not on by default.")
flags.DEFINE_float(
    "gpu_memory_fraction", 0.95,
        "Fraction of GPU memory to allocate.")


###### Train flags
flags.DEFINE_string(
    "input_pipeline_train", "{}", 
        "YAML configuration string for the training data input pipeline.")
flags.DEFINE_string(
    "input_pipeline_dev", "{}", 
        "YAML configuration string for the development data input pipeline.")
flags.DEFINE_integer(
    "batch_size", 16, 
        "Batch size used for training and evaluation.")
flags.DEFINE_string(
    "warm_start_from", None, 
        "Warm start from checkpoint.")

flags.DEFINE_string(
    "schedule", "train_and_evaluate", 
        "Method of Experiment to run.")
flags.DEFINE_integer(
    "log_step_count_steps", 100, 
        "Number of local steps after which progress is printed out")
flags.DEFINE_integer(
    "train_steps", 250000, 
        "The number of steps to run training for.")

# multilingual knowledge distillation
flags.DEFINE_string(
    "multikd_lang_pairs", "", 
        "a list of language pairs separated by comma, E.g. en-zh,en-fr,de-en")
flags.DEFINE_string(
    "multikd_teacher_scores", "", 
        "a list of teacher scores separated by comma, same order with multikd_lang_pairs")
flags.DEFINE_float(
    "multikd_delta", 0., 
        "delta when comparing teacher and student scores, "
        "i.e., continue KD if student_score < teacher_score + delta")

# eval
flags.DEFINE_bool(
    "eval_run_autoregressive", False, 
        "Run eval autoregressively where we condition on previous"
        "generated output instead of the actual target.")
flags.DEFINE_string(
    "early_stopping_metric", "loss", 
        "If --eval_early_stopping_steps is not None, then stop "
        "when --eval_early_stopping_metric has not decreased for "
        "--eval_early_stopping_steps")
flags.DEFINE_float(
    "early_stopping_metric_delta", 0., 
        "Delta determining whether metric has plateaued.")
flags.DEFINE_integer(
    "early_stopping_rounds", None, 
        "If --eval_early_stopping_steps is not None, then stop "
        "when --eval_early_stopping_metric has not decreased for "
        "--eval_early_stopping_steps")
flags.DEFINE_bool(
    "early_stopping_metric_minimize", True, 
        "Whether to check for the early stopping "
        "metric going down or up.")
flags.DEFINE_integer(
    "eval_every_n_steps", 1000,
        "Save checkpoints and run evaluation every N steps during "
        "local training.")
flags.DEFINE_integer(
    "eval_throttle_seconds", 600,
        "Do not re-evaluate unless the last evaluation was started"
        " at least this many seconds ago.")
flags.DEFINE_integer(
    "eval_keep_best_n", 0, "best-n checkpoints to keep by copying to other dir")
flags.DEFINE_string(
    "eval_keep_best_metric", None, 
        "If --eval_early_stopping_steps is not None, then stop "
        "when --eval_early_stopping_metric has not decreased for "
        "--eval_early_stopping_steps")
flags.DEFINE_bool(
    "eval_keep_best_metric_minimize", None, 
        "Whether to check for the early stopping "
        "metric going down or up.")

# checkpoints
flags.DEFINE_integer(
    "keep_checkpoint_max", 50, 
        "How many recent checkpoints to keep.")
flags.DEFINE_integer(
    "keep_checkpoint_every_n_hours", 10000, 
        "Number of hours between each checkpoint to be saved. "
        "The default value 10,000 hours effectively disables it.")
flags.DEFINE_integer(
    "save_checkpoints_secs", 0,
        "Save checkpoints every this many seconds. "
        "Default=0 means save checkpoints each x steps where x "
        "is max(iterations_per_loop, local_eval_frequency).")


###### decode and export
flags.DEFINE_string(
    "tasks", "{}", 
    "List of inference tasks to run.")
flags.DEFINE_string(
    "input_pipeline", None,
        "Defines how input data should be loaded. A YAML string.")
flags.DEFINE_string(
    "checkpoint_path", None,
        "Full path to the checkpoint to be loaded. If None, "
        "the latest checkpoint in the model dir is used.")
flags.DEFINE_string(
    "export_dir", None, 
        "directory to export model to")
flags.DEFINE_string(
    "bleu_script", None,
        "use this script to calc bleu instead. CMD: bash script hyp ref")
flags.DEFINE_string(
    "reference", None,
        "reference file for calculating bleu. provide prefix if multiple references")
flags.DEFINE_bool(
    "sort_input", True, 
        "sort input file according to sentence length (descending)")
flags.DEFINE_bool(
    "init_from_scaffold", False, 
        "init params by scaffold rather than loading local checkpoint. "
        "Used in estimator.predict for ensemble_seq2seq.")
flags.DEFINE_bool(
    "score", False, 
        "if score=True, input both source and target for scoring")


###### multi-gpu/distributed training flags
flags.DEFINE_boolean(
    "data_parallelism", True,
        "use data_parallelism on multiple-gpus, i.e., batch is splited and sent to each gpu. "
        "Otherwise, use model_parallelsim")
flags.DEFINE_boolean(
    "dp_param_shard", True,
        "store params on all devices according to size")
flags.DEFINE_bool(
    "locally_shard_to_cpu", False,
        "Use CPU as a sharding device running locally. This allows "
        "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool(
    "daisy_chain_variables", False,
        "This setting controls whether to copy variables around in a daisy chain "
        "(if true) or leave their placement to TensorFlow. It only affects multi "
        "device training and mostly should be turned on for performance. One "
        "exception are recurrent models: with dynamic loops it must be off.")

flags.DEFINE_string(
    "master", "", 
        "Address of TensorFlow master.")
flags.DEFINE_bool(
    "sync", False, 
        "Sync compute on PS.")
flags.DEFINE_string(
    "worker_job", "/job:localhost", 
        "name of worker job")
flags.DEFINE_integer(
    "worker_gpu", 0,
        "the number of gpus available in a single worker")
flags.DEFINE_integer(
    "worker_replicas", 1, 
        "How many workers to use.")
flags.DEFINE_integer(
    "worker_id", 0, 
        "Which worker task are we.")
flags.DEFINE_integer(
    "ps_gpu", 0, 
        "How many GPUs to use per ps.")
flags.DEFINE_string(
    "gpu_order", "", 
        "Optional order for daisy-chaining GPUs."
        " e.g. \"1 3 2 4\"")
flags.DEFINE_string(
    "ps_job", "/job:ps", 
        "name of ps job")
flags.DEFINE_integer(
    "ps_replicas", 0, 
        "How many ps replicas.")

# horovod
flags.DEFINE_float(
    "gradient_clip", None, 
        "value for gradient_clip")
flags.DEFINE_string(
    "efs_model_dir", None, 
        "The directory to write model checkpoints and summaries to. "
        "If None, a local temporary directory is created.")
flags.DEFINE_bool(
    "clear_efs", False, 
        "whether to clear efs")

# modify for npu overflow start
# enable overflow
flags.DEFINE_string("over_dump", "False",
                    "whether to enable overflow")
flags.DEFINE_string("over_dump_path", "./",
                    "path to save overflow dump files")
# modify for npu overflow end
