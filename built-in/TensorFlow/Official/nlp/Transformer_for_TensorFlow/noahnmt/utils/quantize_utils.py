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

"""Synchronize replicas for model average training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import six


import tensorflow as tf


def clip_matmul_inputs(params, mode):
  tf.logging.info("Add clip_by_value to inputs of MatMul")

  value = tf.constant(float(params["clip_gemm.value"]))
  batch_matmul = params["clip_gemm.batch_matmul"]

  # if mode == tf.estimator.ModeKeys.TRAIN:
  global_step = tf.train.get_global_step()

  decay = tf.train.exponential_decay(
      value,
      (tf.minimum(global_step, params["clip_gemm.stop_decay_at"]) - params["clip_gemm.start_decay_step"]),
      params["clip_gemm.decay_steps"], 
      params["clip_gemm.decay_rate"], 
      staircase=params["clip_gemm.staircase"])
  decay = tf.maximum(decay, params["clip_gemm.min_value"])

  decay_value = tf.cond(
      global_step < params["clip_gemm.start_decay_step"],
      lambda: value,
      lambda: decay)
  # else:
  #   decay_value = value

  visited = {}
  count = 0
  graph = tf.get_default_graph()
  graph_context = graph._get_control_flow_context()
  for op in graph.get_operations():
    if op.type == "MatMul" or (batch_matmul and op.type == "BatchMatMul"):
      inputs = op.inputs
      count += 1
      assert len(inputs) == 2

      for i, inp in enumerate(inputs):
        assert inp.dtype == tf.float32
        if inp.name in visited:
          op._update_input(i, visited[inp.name])
        else:
          with tf.name_scope(op.name + "/input_%d"%i), tf.device(op.device):
            # set context
            context = inp.op._control_flow_context
            if context:
              context.Enter()
            # create op
            clip_inp = tf.clip_by_value(inp, -decay_value, decay_value)
            # reset context
            if context:
              context.Exit()
            # update matmul inputs
            op._update_input(i, clip_inp)
            visited[inp.name] = clip_inp
  tf.logging.info("%d inputs of %d MatMuls are clipped" % (len(visited), count))


def quantize8(clip_value, batch_matmul=False):
  tf.logging.info("Quantize 8-bits to inputs of MatMul")

  visited = {}
  count = 0
  # assert_ops = []
  for op in tf.get_default_graph().get_operations():
    if op.type == "MatMul" or (batch_matmul and op.type == "BatchMatMul"):
      inputs = op.inputs
      count += 1
      unquant_mul = 1.
      # valid = False
      assert len(op.inputs) == 2

      context = op._control_flow_context
      if context:
        context.Enter()

      quant_inputs = []
      for i, inp in enumerate(inputs):
        assert inp.dtype == tf.float32
        # valid = True
        if inp.name in visited:
          quant_inp, quant_mul = visited[inp.name]
        else:
          with tf.name_scope(op.name + "/input_%d"%i), tf.device(op.device):
            # find max value
            max_value = tf.reduce_max(tf.abs(inp))
            quant_mul = 127. / max_value              
            quant_inp = tf.round(inp * quant_mul)
            # check overflow
            quant_inp = tf.cast(tf.cast(quant_inp, tf.int8), dtype=tf.int32)
            quant_inputs.append(quant_inp)
            visited[inp.name] = (quant_inp, quant_mul)
        # update input
        # op._update_input(i, quant_inp)
        unquant_mul *= 1. / quant_mul

      # unquant outputs
      # if valid:
      assert len(quant_inputs) == 2
      if op.type == "MatMul":
        quant_output = tf.matmul(
            quant_inputs[0], quant_inputs[1],
            transpose_a=op.get_attr("transpose_a"), 
            transpose_b=op.get_attr("transpose_b"))
      elif batch_matmul and op.type == "BatchMatMul":
        quant_output = tf.matmul(
            quant_inputs[0], quant_inputs[1],
            adjoint_a=op.get_attr("adj_x"), 
            adjoint_b=op.get_attr("adj_y"))
      else:
        raise ValueError("Unkown op type!")
      

      assert len(op.outputs) == 1
      output = op.outputs[0]
      consumers = output.consumers()
      assert len(consumers) > 0

      # cast for overflow checking
      unquant_output = unquant_mul * tf.cast(quant_output, dtype=tf.float32)
      
      tensors_modified_count = tf.contrib.graph_editor.reroute_ts(
          [unquant_output], [output], can_modify=consumers)
      # Some operations can have multiple output tensors going to the same
      # consumer. Since consumers is a set, we need to ensure that
      # tensors_modified_count is greater than or equal to the length of the set
      # of consumers.
      if tensors_modified_count < len(consumers):
        raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
            [consumer.name for consumer in consumers]))
      
      del output
      del op
      
      if context:
        context.Exit()

  tf.logging.info("%d inputs of %d MatMuls are quantized" % (len(visited), count))


def quantize16(clip_value, batch_matmul=False):
  tf.logging.info("Quantize 16-bits to inputs of MatMul")

  BITS = 10
  if clip_value:
    BITS = int(clip_value)
  quant_mul = float(2 ** BITS)
  unquant_mul = 1. / (quant_mul * quant_mul)

  visited = {}
  count = 0
  for op in tf.get_default_graph().get_operations():
    if op.type == "MatMul" or (batch_matmul and op.type == "BatchMatMul"):
      inputs = op.inputs
      count += 1
      unquant_mul = 1.
      # valid = False
      assert len(op.inputs) == 2

      context = op._control_flow_context
      if context:
        context.Enter()

      quant_inputs = []
      for i, inp in enumerate(inputs):
        assert inp.dtype == tf.float32
        # valid = True
        if inp.name in visited:
          quant_inp = visited[inp.name]
        else:
          with tf.name_scope(op.name + "/input_%d"%i), tf.device(op.device):                          
            quant_inp = tf.round(inp * quant_mul)
            # check overflow
            quant_inp = tf.cast(tf.cast(quant_inp, tf.int16), dtype=tf.int32)
            visited[inp.name] = quant_inp
        # update input
        quant_inputs.append(quant_inp)
        # op._update_input(i, quant_inp)
        # unquant_mul *= 1. / quant_mul

      # unquant outputs
      # if valid:
      assert len(quant_inputs) == 2
      if op.type == "MatMul":
        quant_output = tf.matmul(
            quant_inputs[0], quant_inputs[1],
            transpose_a=op.get_attr("transpose_a"), 
            transpose_b=op.get_attr("transpose_b"))
      elif batch_matmul and op.type == "BatchMatMul":
        quant_output = tf.matmul(
            quant_inputs[0], quant_inputs[1],
            adjoint_a=op.get_attr("adj_x"), 
            adjoint_b=op.get_attr("adj_y"))
      else:
        raise ValueError("Unkown op type!")

      assert len(op.outputs) == 1
      output = op.outputs[0]
      consumers = output.consumers()
      assert len(consumers) > 0

      # cast for overflow checking
      unquant_output = unquant_mul * tf.cast(quant_output, dtype=tf.float32)
      # use graph editor
      tensors_modified_count = tf.contrib.graph_editor.reroute_ts(
          [unquant_output], [output], can_modify=consumers)
      # Some operations can have multiple output tensors going to the same
      # consumer. Since consumers is a set, we need to ensure that
      # tensors_modified_count is greater than or equal to the length of the set
      # of consumers.
      if tensors_modified_count < len(consumers):
        raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
            [consumer.name for consumer in consumers]))
      
      del output
      del op
      
      if context:
        context.Exit()

  tf.logging.info("%d inputs of %d MatMuls are quantized" % (len(visited), count))