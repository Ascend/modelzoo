# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

####################NPU_modify start####################
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu import npu_loss_scale_optimizer as lso
from npu_bridge.estimator.npu import npu_loss_scale_manager as lsm_lib
####################NPU_modify end######################

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, manual_fp16=False, use_tpu=False):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  ####################NPU_modify start####################
  optimizer = NPUDistributedOptimizer(optimizer)
  if tf.flags.FLAGS.npu_bert_loss_scale not in [None, -1]:
    opt_tmp = optimizer
    if tf.flags.FLAGS.npu_bert_loss_scale == 0:
      loss_scale_manager = lsm_lib.ExponentialUpdateLossScaleManager(init_loss_scale=tf.flags.FLAGS.init_loss_scale_value, \
                                                                     incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    elif tf.flags.FLAGS.npu_bert_loss_scale >= 1:
      loss_scale_manager = lsm_lib.FixedLossScaleManager(loss_scale=tf.flags.FLAGS.npu_bert_loss_scale)
    else:
      raise ValueError("Invalid loss scale: %d" % tf.flags.FLAGS.npu_bert_loss_scale)
    optimizer = lso.NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=tf.flags.FLAGS.distributed)
  ####################NPU_modify end######################

  tvars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, tvars)
  # grads = tf.gradients(loss, tvars)

  grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
  grads, tvars = list(zip(*grads_and_vars))

  # This is how the model was pre-trained.
  ####################NPU_modify start####################
  if tf.flags.FLAGS.npu_bert_clip_by_global_norm:
    (clipped_grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  else:
    with tf.name_scope("clip_grads"):
      clipped_grads = [
        (tf.clip_by_norm(grad, clip_norm=1.0))
        if grad is not None else (grad, var) for grad in grads
      ]
  ####################NPU_modify end######################

  with tf.name_scope("apply_grads"):
    train_op = optimizer.apply_gradients(
      list(zip(clipped_grads, tvars)), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.

  ####################NPU_modify start####################
  # new_global_step = global_step + 1
  # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  ####################NPU_modify end######################

  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    ####################NPU_modify start####################
    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    ####################NPU_modify end######################
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, manual_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      with tf.name_scope("apply_one_adam"):
          if grad is None or param is None:
            continue

          param_name = self._get_variable_name(param.name)
          has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
          if has_shadow:
              # create shadow fp32 weights for fp16 variable
              param_fp32 = tf.get_variable(
                  name=param_name + "/shadow",
                  dtype=tf.float32,
                  trainable=False,
                  initializer=tf.cast(param.initialized_value(), tf.float32))
          else:
              param_fp32 = param

          m = tf.get_variable(
              name=param_name + "/adam_m",
              shape=param.shape.as_list(),
              dtype=tf.float32,
              trainable=False,
              initializer=tf.zeros_initializer())
          v = tf.get_variable(
              name=param_name + "/adam_v",
              shape=param.shape.as_list(),
              dtype=tf.float32,
              trainable=False,
              initializer=tf.zeros_initializer())

          # Standard Adam update.
          next_m = (
              tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
          next_v = (
              tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                        tf.square(grad)))

          update = next_m / (tf.sqrt(next_v) + self.epsilon)

          # Just adding the square of the weights to the loss function is *not*
          # the correct way of using L2 regularization/weight decay with Adam,
          # since that will interact with the m and v parameters in strange ways.
          #
          # Instead we want ot decay the weights in a manner that doesn't interact
          # with the m/v parameters. This is equivalent to adding the square
          # of the weights to the loss with plain (non-momentum) SGD.
          if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * param_fp32

          update_with_lr = self.learning_rate * update

          next_param = param_fp32 - update_with_lr

          if has_shadow:
              # cast shadow fp32 weights to fp16 and assign to trainable variable
              param.assign(tf.cast(next_param, param.dtype.base_dtype))
          assignments.extend(
              [param_fp32.assign(next_param),
               m.assign(next_m),
               v.assign(next_v)])

    ####################NPU_modify start####################
    new_global_step = global_step + 1
    new_global_step = tf.identity(new_global_step, name='step_update')
    assignments.extend([global_step.assign(new_global_step)])
    ####################NPU_modify end######################
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
