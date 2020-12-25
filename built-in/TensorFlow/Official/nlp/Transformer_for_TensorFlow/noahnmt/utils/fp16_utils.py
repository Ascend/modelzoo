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

import tensorflow as tf

class DynamicLossScaler:
  def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000):
    self.loss_scale = tf.get_variable(
          initializer=tf.constant_initializer(init_scale),
          trainable=False,
          shape=(),
          dtype=tf.float32,
          name="dls_loss_scale")
    self.scale_factor = tf.constant(scale_factor, dtype=tf.float32)
    self.scale_window = tf.constant(scale_window, dtype=tf.int64)
    self._iter = tf.get_variable(
          initializer=tf.constant_initializer(0),
          trainable=False,
          shape=(),
          dtype=tf.int64,
          name="dls_iter")
    self._last_overflow_iter = tf.get_variable(
          initializer=tf.constant_initializer(-1),
          trainable=False,
          shape=(),
          dtype=tf.int64,
          name="dls_last_overflow_iter")

  def update_scale(self, overflow):
    def _overflow_ops():
      return tf.group(*[
          tf.assign(self.loss_scale, self.loss_scale / self.scale_factor).op,
          tf.assign(self._last_overflow_iter, self._iter).op])

    def _normal_ops():
      def _update():
        return tf.assign(self.loss_scale, self.loss_scale * self.scale_factor).op
      return tf.cond(
                tf.equal(tf.mod(self._iter - self._last_overflow_iter, self.scale_window), 0),
                _update, tf.no_op)
    
    op_ = tf.cond(overflow, _overflow_ops, _normal_ops)

    with tf.control_dependencies([op_]):
      return tf.assign_add(self._iter, 1).op


  @staticmethod
  def has_overflow(grad_norm):
    # detect inf and nan
    return tf.logical_or(
              tf.is_nan(grad_norm),
              tf.not_equal(grad_norm, grad_norm))
    # if grad_norm == float('inf') or grad_norm != grad_norm:
    #   return True
    # return False


class FP16Helper():
  """ TODO
  """
  def __init__(self):
    # dynamically scale loss to reduce overflow
    self.scaler = DynamicLossScaler(init_scale=2.**7)
    self._fp32_params = []
    self._fp16_params = []
    self._init_op = None
    self._update_op = None


  def fp16_loss(self, loss):
    # dynamically rescale loss to stay in FP16 range
    return tf.cast(loss * self.scaler.loss_scale, dtype=tf.float16)
  

  def fp32_variables(self):
    return self._fp32_params

  
  # def create_init_op(self):
  #   init_ops = []
  #   for _, var in gradvars:
  #     fp32_v = tf.get_variable(
  #                 initializer=init_ops.zeros_initializer(var.dtype),
  #                 shape=var.get_shape(),
  #                 trainable=False,
  #                 dtype=tf.float32,
  #                 name="fp32/%s" %var.name)
  #     self._fp32_params.append(fp32_v)
  #     init_ops.append(tf.assign(fp32_v, tf.cast(var, dtype=tf.float32)).op)
  #   return tf.group(*init_ops)
  

  def create_update_op(self):
    def _update_op():
      init_ops = []
      for v16, v32 in zip(self._fp16_params, self._fp32_params):
        with tf.device(v16.device):
          init_ops.append(tf.assign(v16, tf.cast(v32, dtype=tf.float16)).op)
      return tf.group(*init_ops)
    # if overflow, don't assign
    return tf.cond(self._overflow, tf.no_op, _update_op)
  

  def fp32_gradvars(self, gradvars, max_norm=None):
    gradients = []
    for grad, var in gradvars:
      gradients.append(grad)
      with tf.device(var.device):
        # create fp32 variables
        fp32_v = tf.get_variable(
                    initializer = tf.cast(var.initialized_value(), dtype=tf.float32),
                    trainable=False,
                    dtype=tf.float32,
                    name="fp32_helper/%s" % var.op.name)
      self._fp32_params.append(fp32_v)
      self._fp16_params.append(var)

    # conver gradient to fp32
    grads_fp32 = []
    for grad in gradients:
      if isinstance(grad, tf.Tensor):
        grad = tf.cast(grad, dtype=tf.float32)
      elif isinstance(grad, tf.IndexedSlices):
        grad = tf.IndexedSlices(
            values=tf.cast(grad.values, dtype=tf.float32),
            indices=grad.indices,
            dense_shape=grad.dense_shape)
      else:
        raise ValueError("unknow gradient type")
      # undo effect of dynamic loss scaling on gradients
      # rescale
      grad = grad / self.scaler.loss_scale

      grads_fp32.append(grad)
    gradients = grads_fp32

    # undo effect of dynamic loss scaling on gradients
    # rescale
    # gradients = [grad / self.scaler.loss_scale for grad in gradients]
    
    if max_norm:
      gradients, grad_norm = tf.clip_by_global_norm(gradients, max_norm)
    else:
      grad_norm = tf.global_norm(gradients)
    
    # detect overflow and adjust loss scale
    overflow = DynamicLossScaler.has_overflow(grad_norm)
    self._overflow = overflow
    update_scale_op = self.scaler.update_scale(overflow)
    
    with tf.control_dependencies([update_scale_op]):
      # if overflow, zero grads
      mask = 1. - tf.cast(overflow, dtype=tf.float32)
      gradients = [g * mask for g in gradients]

    return zip(gradients, self._fp32_params)
