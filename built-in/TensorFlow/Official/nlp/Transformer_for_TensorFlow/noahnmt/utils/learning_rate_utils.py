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

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import tensorflow as tf


def learning_rate_factor(
        name, step_num, params, 
        train_steps=None, 
        hidden_size=None, 
        num_replicas=1):
  """Compute the designated learning rate factor from hparams."""
  warmup_steps = float(params["learning_rate.warmup_steps"])
  start_decay_step = float(params["learning_rate.start_decay_step"])
  stop_decay_step = float(params["learning_rate.stop_decay_at"])

  if name == "constant":
    tf.logging.info("Base learning rate: %f", params["learning_rate.constant"])
    return params["learning_rate.constant"]
  elif name == "linear_warmup":
    return tf.minimum(1.0, step_num / params["learning_rate.warmup_steps"])
  elif name == "linear_decay":
    ret = (train_steps - step_num) / params["learning_rate.decay_steps"]
    return tf.minimum(1.0, tf.maximum(0.0, ret))
  elif name == "rsqrt_decay":
    return tf.rsqrt(tf.maximum(step_num, params["learning_rate.warmup_steps"]))
  elif name == "rsqrt_normalized_decay":
    scale = tf.sqrt(tf.to_float(params["learning_rate.warmup_steps"]))
    return scale * tf.rsqrt(tf.maximum(
        step_num, params["learning_rate.warmup_steps"]))
  elif name == "exp_decay":
    decay_steps = params["learning_rate.decay_steps"]
    warmup_steps = params["learning_rate.warmup_steps"]
    p = (step_num - warmup_steps) / decay_steps
    p = tf.maximum(p, 0.)
    if params["learning_rate.decay_staircase"]:
      p = tf.floor(p)
    return tf.pow(params["learning_rate.decay_rate"], p)
  elif name == "rnmt_warmup_decay":
    score1 = 1. + step_num * (num_replicas - 1) / (num_replicas * warmup_steps)
    score2 = float(num_replicas)
    score3 = num_replicas * (2 * num_replicas) ** ((start_decay_step - num_replicas * step_num)/(stop_decay_step - start_decay_step))
    return tf.minimum(tf.minimum(score1, score2), score3)
  elif name == "rsqrt_hidden_size":
    return hidden_size ** -0.5
  else:
    raise ValueError("unknown learning rate factor %s" % name)


def learning_rate_schedule(params, hparams, num_replicas):
  """Learning rate schedule based on hparams."""
  step_num = _global_step(params)
  schedule_string = params["learning_rate.schedule"]
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= learning_rate_factor(
        name, step_num, params, 
        hparams.train_steps, 
        params["embedding.dim"],
        num_replicas)
  return ret


def _global_step(params):
  """Adjust global step if a multi-step optimizer is used."""
  step = tf.to_float(tf.train.get_or_create_global_step())
  multiplier = params["optimizer.params"]["accumulate_steps"]
  if not multiplier or params["optimizer.name"] != "MultistepAdam":
    return step

  tf.logging.info("Dividing global step by %d for multi-step optimizer."
                  % multiplier)
  return step / tf.to_float(multiplier)