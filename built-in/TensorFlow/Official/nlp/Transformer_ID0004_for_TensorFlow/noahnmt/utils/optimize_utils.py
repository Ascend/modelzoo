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

"""optimize utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six
import os

import tensorflow as tf
import pdb

from noahnmt.layers import common_layers as common_utils
from noahnmt.utils import graph_utils

# from noahnmt.optimizers.multistep_optimizer import MultistepAdamOptimizer

# from tensorflow.contrib.offline_train.python.npu.npu_optimizer import NPUDistributedOptimizer
# from tensorflow.contrib.offline_train.python.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
# from tensorflow.contrib.offline_train.python.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager

RANK_SIZE = int(os.environ.get('RANK_SIZE', '1').strip())
RANK_ID = int(os.environ.get('DEVICE_ID', '0').strip())

class DynamicLossScaleManager(tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager):
    def update_loss_scale(self, finite_grads):
        """Updates loss scale based on if gradients are finite in current step."""
    
        max_scale = float(2 ** 14)

        def update_if_finite_grads():
            """Branch function when grads are all finite."""

            def incr_loss_scale():
                new_loss_scale = tf.cond(
                    self._loss_scale * self._incr_ratio,
                    lambda: self._loss_scale * self._incr_ratio,
                    lambda: self._loss_scale)
                # new_loss_scale = tf.minimum(new_loss_scale, max_scale)
                print("no max limit"*100)
                update_op = tf.assign(self._loss_scale, new_loss_scale)
                # When loss_scale is updated, both good and bad steps are reset.
                return tf.group(update_op, self._reset_stats())

            return tf.cond(
              self._num_good_steps + 1 >= self._incr_every_n_steps,
              incr_loss_scale,
              lambda: tf.assign_add(self._num_good_steps, 1).op)

        def update_if_not_finite_grads():
            """Branch function when any grad is not finite."""

            def decr_loss_scale():
                update_op = tf.assign(
                    self._loss_scale,
                    tf.maximum(1., self._loss_scale * self._decr_ratio))
                # When loss_scale is updated, both good and bad steps are reset.
                return tf.group(update_op, self._reset_stats())

            def just_update_steps():
                # When bad_steps is incremented, good_step is reset.
                return tf.group(
                    tf.assign_add(self._num_bad_steps, 1),
                    tf.assign(self._num_good_steps, 0))

            return tf.cond(
              self._num_bad_steps + 1 >= self._decr_every_n_nan_or_inf,
              decr_loss_scale, just_update_steps)

        return tf.cond(finite_grads, update_if_finite_grads,
                                     update_if_not_finite_grads)


def optimize(loss, learning_rate, params, hparams=None, mixed_precision=False, mixed_precision_params=None, is_finite=None):
    """Minimize loss."""
    local_loss = loss

    opt = ConditionalOptimizer(learning_rate, params, hparams)
    opt = NPUDistributedOptimizer(opt)

    if mixed_precision:
        tf.logging.info("Using mixed precision")
        if mixed_precision_params["fix_loss_scale"]:
            loss_scale_manager = FixedLossScaleManager(mixed_precision_params["init_loss_scale"])
        else:
            loss_scale_manager = ExponentialUpdateLossScaleManager(
                init_loss_scale=mixed_precision_params["init_loss_scale"],
                incr_every_n_steps=mixed_precision_params["incr_every_n_steps"],
                decr_every_n_nan_or_inf=mixed_precision_params["decr_every_n_nan_or_inf"],
                incr_ratio=mixed_precision_params["incr_ratio"],
                decr_ratio=mixed_precision_params["decr_ratio"])
        loss_scale = loss_scale_manager.get_loss_scale()
        tf.summary.scalar("loss_scale", loss_scale)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=RANK_SIZE>1)

    gradvars = opt.compute_gradients(
        local_loss,
        colocate_gradients_with_ops=True)

    out_dict = {}
    for grad, var in gradvars:
        tf.logging.info(var.name)
        assert grad is not None
        out_dict[var.name + "_grad"] = grad
    graph_utils.add_dict_to_collection(out_dict, "SAVE_TENSOR")
    

    if params["max_grad_norm"]:
        tf.logging.info("Clipping gradients, norm: %0.5f", params["max_grad_norm"])
        grads, vars_ = zip(*gradvars)
        grads = clip_gradients(grads, params["max_grad_norm"], mixed_precision)
        #tf.summary.histogram("gradients", grads)
        #summarize_variables(grads,"grads")
        #summarize_variables(vars_,"vars_")
        gradvars = list(zip(grads, vars_))

    # Create single grouped train op
    global_step = tf.train.get_global_step()
    
    #train_op = opt.apply_gradients(gradvars)
    #with tf.control_dependencies([train_op]):
    #     train_op = tf.assign_add(global_step, 1)

    # def true_apply_gradients_fn():
    #     return opt.apply_gradients(gradvars, global_step=global_step)

    # train_op = tf.cond(is_finite,
    #                     true_apply_gradients_fn,
    #                     tf.no_op)

    train_op = opt.apply_gradients(gradvars, global_step=global_step)

    return train_op, []


def clip_gradients(grads, norm=None, mixed_precision=False):
    """Clips gradients by global norm."""
    clipped_grads = [tf.clip_by_norm(grad, norm) for grad in grads]
    return clipped_grads
    
    if not mixed_precision:
        clipped_grads, _ = tf.clip_by_global_norm(
            grads, norm)
    else:
        all_are_finite = tf.reduce_all([tf.reduce_all(g) for g in grads])
        # to prevent clip_by_global_norm from having a hizzy fit.

        clipped_grads, _ = tf.clip_by_global_norm(
            grads, norm,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)))
        
    return clipped_grads


class ConditionalOptimizer(tf.train.Optimizer):
    """Conditional optimizer."""

    def __init__(self, lr, params, hparams=None):  # pylint: disable=super-init-not-called
        optimizer_name = params["optimizer.name"]
        optimizer_params = params["optimizer.params"]
        tf.logging.info("Using optimizer %s", optimizer_name)
        self._name_for_cast = optimizer_name

        if optimizer_name == "SGD":
            self._opt = tf.train.GradientDescentOptimizer(lr)
        elif optimizer_name == "Adam":
            # We change the default epsilon for Adam.
            self._opt = tf.train.AdamOptimizer(
                lr,
                beta1=optimizer_params["beta1"],
                beta2=optimizer_params["beta2"],
                epsilon=optimizer_params["epsilon"])
        elif optimizer_name == "LazyAdam":
            # Using LazyAdam as it's much faster for large vocabulary embeddings.
            self._opt = tf.contrib.opt.LazyAdamOptimizer(
                lr,
                beta1=optimizer_params["beta1"],
                beta2=optimizer_params["beta2"],
                epsilon=optimizer_params["epsilon"])
        elif optimizer_name == "Momentum":
            self._opt = tf.train.MomentumOptimizer(
                lr,
                momentum=optimizer_params["momentum"],
                use_nesterov=optimizer_params["use_nesterov"])
        else:
            self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

    def compute_gradients(self, loss, var_list=None, **kwargs):  # pylint: disable=arguments-differ
        gradients = self._opt.compute_gradients(loss, var_list, **kwargs)
        if self._name_for_cast == "MultistepAdam":
            def cast_grad(g, v):
                if v is not None and g is not None:
                    g = common_utils.cast_like(g, v)
                return (g, v)

            gradients = [cast_grad(g, v) for g, v in gradients]
        return gradients

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        return self._opt.apply_gradients(
            grads_and_vars, global_step=global_step, name=name)


def log_variable_sizes(var_list=None, tag=None, verbose=False):
    """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
  """
    if var_list is None:
        var_list = tf.trainable_variables()
    if tag is None:
        tag = "Trainable Variables"

    if not var_list:
        return

    name_to_var = {v.name: v for v in var_list}
    total_size = 0
    for v_name in sorted(list(name_to_var)):
        v = name_to_var[v_name]
        v_size = int(np.prod(np.array(v.shape.as_list())))
        if verbose:
            tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                            v.name[:-2].ljust(80),
                            str(v.shape).ljust(20), v_size)
        total_size += v_size
    tf.logging.info("%s Total size: %d", tag, total_size)


def summarize_variables(var_list=None, tag=None):
    """Summarize the variables.

  Args:
    var_list: a list of variables; defaults to trainable_variables.
    tag: name scope of the summary; defaults to training_variables/.
  """
    if var_list is None:
        var_list = tf.trainable_variables()
    if tag is None:
        tag = "training_variables/"

    name_to_var = {v.name: v for v in var_list}
    for v_name in list(name_to_var):
        v = name_to_var[v_name]
        tf.summary.histogram(tag + v_name, v)


def get_variable_initializer(initializer, initializer_gain):
    """Get variable initializer from hparams."""
    if not initializer:
        return None

    if initializer == "use_separate_init":
      # IMPORTANT: currently only used for testing RNMT model
      # i.e., only support 'transformer_encoder' and 'rnmt_decoder'
      # separate initializers can be specified in corresponding coders.
      tf.logging.info("Using separate initializers for encoder and decoder")
      return None

    if not tf.contrib.eager.in_eager_mode():
        tf.logging.info("Using variable initializer: %s", initializer)
    if initializer == "orthogonal":
        return tf.orthogonal_initializer(gain=initializer_gain)
    elif initializer == "uniform":
        max_val = initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif initializer == "trunc_normal":
        return tf.truncated_normal_initializer(stddev=initializer_gain)
    elif initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(
            initializer_gain, mode="fan_avg", distribution="normal")
    elif initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(
            initializer_gain, mode="fan_avg", distribution="uniform")
    elif initializer == "xavier":
        return tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError("Unrecognized initializer: %s" % initializer)
