# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
#

"""Ops for collective operations implemented using hccl."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numbers
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.eager import context

from npu_device import gen_npu_ops


DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1
def LARSV2(input_weight,
           input_grad,
           weight_decay,
           learning_rate,
           hyperpara=0.001,
           epsilon=0.00001,
           use_clip=False,
           name=None):
    if context.executing_eagerly():
        raise RuntimeError("tf.LARSV2() is not compatible with "
                           "eager execution.")

    return gen_npu_ops.lars_v2(input_weight=input_weight,
                               input_grad=input_grad,
                               weight_decay=weight_decay,
                               learning_rate=learning_rate,
                               hyperpara=hyperpara,
                               epsilon=epsilon,
                               use_clip=use_clip,
                               name=name)


def _truncate_seed(seed):
  return seed % _MAXINT32  # Truncate to fit into 32-bit integer

def get_seed(op_seed):
  global_seed = ops.get_default_graph().seed

  if global_seed is not None:
    if op_seed is None:
      op_seed = ops.get_default_graph()._last_id

    seeds = _truncate_seed(global_seed), _truncate_seed(op_seed)
  else:
    if op_seed is not None:
      seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
    else:
      seeds = None, None
  # Avoid (0, 0) as the C++ ops interpret it as nondeterminism, which would
  # be unexpected since Python docs say nondeterminism is (None, None).
  if seeds == (0, 0):
    return (0, _MAXINT32)
  return seeds

def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape

def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """The gradient for `gelu`.

    Args:
        x: A tensor with type is float.
        keep_prob: A tensor, float, rate of every element reserved.
        noise_shape: A 1-D tensor, with type int32, shape of keep/drop what random
            generated.
        seed: Random seed.
        name: Layer name.

    Returns:
        A tensor.
    """
    if context.executing_eagerly():
      raise RuntimeError("tf.dropout() is not compatible with "
                        "eager execution.")
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    if isinstance(keep_prob, float) and keep_prob == 1:
      return x
    seed, seed2 = get_seed(seed)
    noise_shape = _get_noise_shape(x, noise_shape)
    gen_out = gen_npu_ops.drop_out_gen_mask(noise_shape, keep_prob, seed, seed2, name)
    result = gen_npu_ops.drop_out_do_mask(x, gen_out, keep_prob, name)
    return result

@ops.RegisterGradient("DropOutDoMask")
def _DropOutDoMaskGrad(op, grad):
    result = gen_npu_ops.drop_out_do_mask(grad, op.inputs[1],  op.inputs[2])
    return [result, None, None]

def basic_lstm_cell(x, h, c, w, b, keep_prob, forget_bias, state_is_tuple,
                    activation, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.basic_lstm_cell() is not compatible with "
                        "eager execution.")
    x = ops.convert_to_tensor(x, name="x")
    h = ops.convert_to_tensor(h, name="h")
    c = ops.convert_to_tensor(c, name="c")
    w = ops.convert_to_tensor(w, name="w")
    b = ops.convert_to_tensor(b, name="b")
    result = gen_npu_ops.basic_lstm_cell(x, h, c, w, b, keep_prob, forget_bias, state_is_tuple,
                    activation, name)
    return result

@ops.RegisterGradient("BasicLSTMCell")
def basic_lstm_cell_grad(op, dct, dht, dit, djt, dft, dot, dtanhct):

    dgate, dct_1 = gen_npu_ops.basic_lstm_cell_c_state_grad(op.inputs[2], dht, dct, op.outputs[2], op.outputs[3], op.outputs[4], op.outputs[5], op.outputs[6], forget_bias=op.get_attr("forget_bias"), activation=op.get_attr("activation"))
    dw, db = gen_npu_ops.basic_lstm_cell_weight_grad(op.inputs[0], op.inputs[1], dgate)
    dxt, dht = gen_npu_ops.basic_lstm_cell_input_grad(dgate, op.inputs[3], keep_prob=op.get_attr("keep_prob"))

    return [dxt, dht, dct_1, dw, db]

def adam_apply_one_assign(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, add2_y, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.adam_apply_one_assign() is not compatible with "
                        "eager execution.")
    result = gen_npu_ops.adam_apply_one_assign(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, add2_y,name)
    return result

def adam_apply_one_with_decay_assign(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.adam_apply_one_with_decay_assign() is not compatible with "
                        "eager execution.")
    result = gen_npu_ops.adam_apply_one_with_decay_assign(input0, input1, input2, input3, input4,
                   mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y, name)
    return result

@ops.RegisterGradient("DynamicGruV2")
def dynamic_gru_v2_grad(op, dy, doutput_h, dupdate, dreset, dnew, dhidden_new):
    (x, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, init_h) = op.inputs
    (y, output_h, update, reset, new, hidden_new) = op.outputs
    (dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev) = gen_npu_ops.dynamic_gru_v2_grad(x, weight_input, weight_hidden, y, init_h, output_h, dy, doutput_h, update, reset, new, hidden_new, direction=op.get_attr("direction"), cell_depth=op.get_attr("cell_depth"), keep_prob=op.get_attr("keep_prob"), cell_clip=op.get_attr("cell_clip"), num_proj=op.get_attr("num_proj"), time_major=op.get_attr("time_major"), gate_order=op.get_attr("gate_order"), reset_after=op.get_attr("reset_after"))

    return (dx, dw_input, dw_hidden, db_input, db_hidden, seq_length, dh_prev)

@ops.RegisterGradient("DynamicRnn")
def dynamic_rnn_grad(op, dy, dh, dc, di, dj, df, do, dtanhc):
    (x, w, b, seq_length, init_h, init_c) = op.inputs
    (y, output_h, output_c, i, j, f, o, tanhc) = op.outputs
    (dw, db, dx, dh_prev, dc_prev) = gen_npu_ops.dynamic_rnn_grad(x, w, b, y, init_h[-1], init_c[-1], output_h, output_c, dy, dh[-1], dc[-1], i, j, f, o, tanhc, cell_type=op.get_attr("cell_type"), direction=op.get_attr("direction"), cell_depth=op.get_attr("cell_depth"), use_peephole=op.get_attr("use_peephole"), keep_prob=op.get_attr("keep_prob"), cell_clip=op.get_attr("cell_clip"), num_proj=op.get_attr("num_proj"), time_major=op.get_attr("time_major"), forget_bias=op.get_attr("forget_bias"))

    return (dx, dw, db, seq_length, dh_prev, dc_prev)

def lamb_apply_optimizer_assign(input0,input1,input2,input3,mul0_x,mul1_x,mul2_x,
                                mul3_x,add2_y,steps,do_use_weight,weight_decay_rate,name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_optimizer_assign() is not compatible with eager execution")
    update,nextv,nextm=gen_npu_ops.lamb_apply_optimizer_assign(input0,input1,input2,input3,mul0_x,mul1_x,mul2_x,
                                mul3_x,add2_y,steps,do_use_weight,weight_decay_rate,name)
    return update,nextv,nextm

def lamb_apply_weight_assign(input0,input1,input2,input3,input4,name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_weight_assign() is not compatible with eager execution")
    result = gen_npu_ops.lamb_apply_weight_assign(input0,input1,input2,input3,input4,name)
    return result

def dropout_v3(x, keep_prob, noise_shape=None, seed=None, name=None):
  """ The gradient for gelu

  Args:
    x: A tensor with type is float
    keep_prob: A tensor, float, rate of every element reserved
    noise_shape: A 1-D tensor, with type int32, shape of keep/drop what random generated.
    seed: Random seed.
    name: Layer name.
  
  Returns:
    A tensor.
  """
  x = ops.convert_to_tensor(x,name="x")
  if not x.dtype.is_floating:
    raise ValueError("x has to be a floating point tensor since it's going to be scaled. Got a %s tensor instead." % x.dtype)
  
  if isinstance(keep_prob,numbers.Real) and not 0 < keep_prob <=1:
    raise ValueError("Keep_prob must be a scalar tensor or a float in the range (0,1], got %g" % keep_prob)
  
  if isinstance(keep_prob,float) and keep_prob==1:
    return x
  
  seed, seed2 = get_seed(seed)
  noise_shape = _get_noise_shape(x,noise_shape)
  gen_out = gen_npu_ops.drop_out_gen_mask_v3(noise_shape,keep_prob,seed,seed2,name)
  result = gen_npu_ops.drop_out_do_mask_v3(x, gen_out, keep_prob, name)
  return result

@ops.RegisterGradient("DropOutDoMaskV3")
def _DropOutDoMaskV3Grad(op,grad):
  result = gen_npu_ops.drop_out_do_mask_v3(grad, op.inputs[1], op.inputs[2])
  return [result, None, None]