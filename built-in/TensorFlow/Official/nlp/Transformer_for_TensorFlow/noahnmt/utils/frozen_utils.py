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

import re
import copy
import numpy as np
import hashlib

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib as optlib
from tensorflow.python.framework.graph_util import extract_sub_graph
from tensorflow.python.framework import op_def_registry
from tensorflow.core.framework import types_pb2


RegisteredOps = op_def_registry.get_registered_ops()

QUANT_DTYPE = tf.quint8
DT_QUINT8 = tf.quint8
DT_QINT32 = tf.qint32
DT_INT32 = tf.int32
DT_FLOAT = tf.float32

QUANT_MODE = b"MIN_FIRST"
CONTIGUOUS_MIN_MAX = 0
SEPARATE_MIN_MAX = 1

QUANT_OP_SET = {
  "Add": {
      # which attributes to copy directly over.
      "attrs_to_copy": {}, 
      # Extra data type attributes we need to set.
      "dtypes_to_set": {"T1": DT_QUINT8, "T2": DT_QUINT8, "Toutput": DT_QINT32},
      # What depth of inputs the op can read in.
      "input_bit_depth": DT_QUINT8, 
      # The depth of the op's quantized outputs.
      "output_bit_depth": DT_QINT32, 
      # Which inputs (e.g. shapes) aren't involved in the quantization process.
      "unquantized_inputs": {},
      # How the outputs are arranged, either
      # [input0, input1, min0, max0, min1, max1] for contiguous, or
      # [input0, input1, min0, min1, max0, max1] for separate.
      # The separate order is needed because it's the only way to specify unknown
      # numbers of inputs for ops like Concat.
      "min_max_order": CONTIGUOUS_MIN_MAX},
  "AvgPool": {
      "attrs_to_copy": {"ksize", "strides", "padding"}, 
      "dtypes_to_set": {"T": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "BiasAdd": {
      "attrs_to_copy": {}, 
      "dtypes_to_set": {"T1": DT_QUINT8, "T2": DT_QUINT8, "out_type": DT_QINT32}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QINT32, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "Concat": {
      "attrs_to_copy": {"N"}, 
      "dtypes_to_set": {"T": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {0}, 
      "min_max_order": SEPARATE_MIN_MAX}, 
  "Conv2D": {
      "attrs_to_copy": {"strides", "padding"}, 
      "dtypes_to_set": {"Tinput": DT_QUINT8, "Tfilter": DT_QUINT8, "out_type": DT_QINT32}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QINT32, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "MatMul": {
      "attrs_to_copy": {"transpose_a", "transpose_b"}, 
      "dtypes_to_set": {"T1": DT_QUINT8, "T2": DT_QUINT8, "Toutput": DT_QINT32}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QINT32, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "MaxPool": {
      "attrs_to_copy": {"ksize", "strides", "padding"}, 
      "dtypes_to_set": {"T": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "Mul": {
      "attrs_to_copy": {}, 
      "dtypes_to_set": {"T1": DT_QUINT8, "T2": DT_QUINT8, "Toutput": DT_QINT32}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QINT32, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "Relu": {
      "attrs_to_copy": {}, 
      "dtypes_to_set": {"Tinput": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "ResizeBilinear": {
      "attrs_to_copy": {"align_corners"}, 
      "dtypes_to_set": {"T": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {1}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "Relu6": {
      "attrs_to_copy": {}, 
      "dtypes_to_set": {"Tinput": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {}, 
      "min_max_order": CONTIGUOUS_MIN_MAX}, 
  "Reshape": {
      "attrs_to_copy": {}, 
      "dtypes_to_set": {"T": DT_QUINT8}, 
      "input_bit_depth": DT_QUINT8, 
      "output_bit_depth": DT_QUINT8, 
      "unquantized_inputs": {1}, 
      "min_max_order": CONTIGUOUS_MIN_MAX},
}

def replace_matmul_sparse(input_graph_def):
  replaced_graph_def = tf.GraphDef()
  for node in input_graph_def.node:
    if node.op == "MatMul":
      placeholder_node = copy.deepcopy(node)
      placeholder_node.op = "SparseMatMul"
      
      # tensor attr
      if "T" in placeholder_node.attr:
        placeholder_node.attr["Ta"].CopyFrom(
            placeholder_node.attr["T"])
        placeholder_node.attr["Tb"].CopyFrom(
            placeholder_node.attr["T"])
        placeholder_node.attr.pop("T")
      else:
        assert "Ta" in placeholder_node.attr
        assert "Tb" in placeholder_node.attr
      
      # add sparse attr
      placeholder_node.attr["a_is_sparse"].CopyFrom(
          tf.AttrValue(b=False))
      placeholder_node.attr["b_is_sparse"].CopyFrom(
          tf.AttrValue(b=True))

      replaced_graph_def.node.extend([placeholder_node])
    else:
      replaced_graph_def.node.extend([copy.deepcopy(node)])

  return replaced_graph_def


def remove_training_nodes(input_graph, protected_nodes=None):
  """Prunes out nodes that aren't needed for inference.

  There are nodes like Identity and CheckNumerics that are only useful
  during training, and can be removed in graphs that will be used for
  nothing but inference. Here we identify and remove them, returning an
  equivalent graph. To be specific, CheckNumerics nodes are always removed, and
  Identity nodes that aren't involved in control edges are spliced out so that
  their input and outputs are directly connected.

  Args:
    input_graph: Model to analyze and prune.
    protected_nodes: An optional list of names of nodes to be kept
      unconditionally. This is for example useful to preserve Identity output
      nodes.

  Returns:
    A list of nodes with the unnecessary ones removed.
  """
  if not protected_nodes:
    protected_nodes = []

  types_to_remove = {"CheckNumerics": True}

  input_nodes = input_graph.node
  names_to_remove = {}
  for node in input_nodes:
    if node.op in types_to_remove and node.name not in protected_nodes:
      names_to_remove[node.name] = True

  nodes_after_removal = []
  for node in input_nodes:
    if node.name in names_to_remove:
      continue
    new_node = tf.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_remove:
        continue
      new_node.input.extend([full_input_name])
    nodes_after_removal.append(new_node)
  
  # added by Liangyou Li
  # to avoid errors in while loop
  control_nodes = {}
  for node in nodes_after_removal:
    for input_name in node.input:
      if re.match(r"^\^", input_name):
        input_name = node_name_from_input(input_name)
        control_nodes[input_name] = True
    

  types_to_splice = {"Identity": True}
  names_to_splice = {}
  for node in nodes_after_removal:
    if node.op in types_to_splice \
       and node.name not in protected_nodes \
       and node.name not in control_nodes:
      # We don't want to remove nodes that have control edge inputs, because
      # they might be involved in subtle dependency issues that removing them
      # will jeopardize.
      has_control_edge = False
      for input_name in node.input:
        if re.match(r"^\^", input_name):
          has_control_edge = True
      if not has_control_edge:
        assert len(node.input) == 1
        names_to_splice[node.name] = node.input[0]

  nodes_after_splicing = []
  for node in nodes_after_removal:
    if node.name in names_to_splice:
      continue
    new_node = tf.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      while input_name in names_to_splice:
        full_input_name = names_to_splice[input_name]
        input_name = re.sub(r"^\^", "", full_input_name)
      new_node.input.extend([full_input_name])
    assert len(input_before_removal) == len(new_node.input)
    nodes_after_splicing.append(new_node)

  output_graph = tf.GraphDef()
  output_graph.node.extend(nodes_after_splicing)
  return output_graph

# not used currently
def _FloatTensor2Quantized(
    tensor, min_value, max_value, dtype=QUANT_DTYPE, mode="min_first"):
    if mode == "min_combined":
      # out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
      # if T == qint8, out[i] -= (range(T) + 1) / 2.0
      # here range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()
      range_t = float(dtype.max - dtype.min)
      tensor = (tensor - min_value) * range_t / (max_value - min_value)
      if dtype == tf.qint8:
        tensor = tensor - (range_t + 1) / 2.0
      return np.round(tensor)
    elif mode == "min_first":
      # num_discrete_values = 1 << (# of bits in T)
      # range_adjust = num_discrete_values / (num_discrete_values - 1)
      # range = (range_max - range_min) * range_adjust
      # range_scale = num_discrete_values / range
      # quantized = round(input * range_scale) - round(range_min * range_scale) +
      #   numeric_limits<T>::min()
      # quantized = max(quantized, numeric_limits<T>::min())
      # quantized = min(quantized, numeric_limits<T>::max())
      num_discrete_values = float(1 << (dtype.size * 8))
      range_adjust = num_discrete_values / (num_discrete_values - 1)
      range_value = (max_value - min_value) * range_adjust
      range_scale = num_discrete_values / range_value
      tensor = np.round(tensor * range_scale) - np.round(min_value * range_scale) + dtype.min
      tensor = np.maximum(tensor, dtype.min)
      tensor = np.minimum(tensor, dtype.max)
      return tensor
    else:
      raise ValueError("Unknown quantize mode!")


def create_constant_node(name, value, dtype, shape=None):
  node = tf.NodeDef()
  node.op = "Const"
  node.name = name
  node.attr["dtype"].CopyFrom(
      tf.AttrValue(type=dtype.as_datatype_enum))
  node.attr["value"].CopyFrom(
      tf.AttrValue(tensor=tf.make_tensor_proto(
          value, dtype, shape)))
  return node


def should_ignore(ignore_names, name):
  for n in ignore_names:
    if len(n) > 0 and n in name:
      return True
  return False


def quantize_weights(input_graph_def, min_size, ignore_names):
  output_graph_def = tf.GraphDef()
  for node in input_graph_def.node:
    if node.op in {"Const"} and not should_ignore(ignore_names, node.name):
      if "dtype" not in node.attr or "value" not in node.attr:
        raise ValueError("dtype or value not found in node.attr")
      weight = optlib.values_from_const(node)
      if node.attr["dtype"].type != DT_FLOAT.as_datatype_enum \
         or np.size(weight) < min_size:
        output_graph_def.node.extend([copy.deepcopy(node)])
        continue
      # if a const node has inputs, it must be control inputs
      # while loop case
      for input_name in node.input:
        if not input_name.startswith("^"):
          raise ValueError("Const input is not a control input")
      
      min_value = np.min(weight)
      max_value = np.max(weight)
      # Make sure the quantization range includes 0.0f. Not all quantized
      # Ops behave properly if 0.0f is not in the range.
      min_value = min([min_value, 0.0])
      max_value = max([max_value, 0.0])
      # min_value == max_value is a tricky case. It can occur for general
      # tensors, and of course for scalars. The quantized ops cannot deal
      # with this case, so we set max_value to something else.
      # It's a tricky question what is the numerically best solution to
      # deal with this degeneracy.
      if min_value == max_value:
        if abs(min_value) < 0.000001:
          max_value = min_value + 1.0
        elif min_value > 0.:
          max_value = 2 * min_value
        else:
          max_value = min_value / 2.0
      
      sess = tf.Session()
      with sess.as_default():
        quantize_op = tf.quantize(
            weight,
            min_value,
            max_value,
            QUANT_DTYPE,
            mode=QUANT_MODE)
        quantized_weight = quantize_op[0].eval()

      quantized_node = create_constant_node(
          node.name+"_quantized_const", quantized_weight, QUANT_DTYPE)
      # add control input to fix bugs in while loop
      quantized_node.input.extend(node.input)
      output_graph_def.node.extend([quantized_node])
      min_node = create_constant_node(
          node.name+"_quantized_min", min_value, DT_FLOAT)
      # add control input to fix bugs in while loop
      min_node.input.extend(node.input)
      output_graph_def.node.extend([min_node])
      max_node = create_constant_node(
          node.name+"_quantized_max", max_value, DT_FLOAT)
      # add control input to fix bugs in while loop
      max_node.input.extend(node.input)
      output_graph_def.node.extend([max_node])

      dequantized_node = create_node(
          "Dequantize", node.name, 
          [quantized_node.name, min_node.name, max_node.name])
      set_attr_dtype(dequantized_node, "T", QUANT_DTYPE)
      set_attr_string(dequantized_node, "mode", QUANT_MODE)
      output_graph_def.node.extend([dequantized_node])
    else:
      output_graph_def.node.extend([copy.deepcopy(node)])
  optlib.ensure_graph_is_valid(output_graph_def)
  return output_graph_def


def node_name_parts_from_input(input_name):
  input_parts = input_name.split(":")
  
  if len(input_parts) < 2:
    suffix = ""
  else:
    suffix = ":" + input_parts[1]

  node_name = input_parts[0]
  if input_parts[0].startswith("^"):
    prefix = "^"
    node_name = node_name[1:]
  else:
    prefix = ""
  
  return prefix, node_name, suffix

# Replaces invalid characters in input names to get a unique node name.
def unique_node_name_from_input(input_name):
  return input_name.replace(":", "__port__").replace("^", "__hat__")
  # prefix, node_name, suffix = node_name_parts_from_input(input_name)
  # result = ""
  # if prefix == "^":
  #   result += "__hat__"
  # result += node_name
  # if len(suffix) > 0:
  #   result += "__port__" + suffix[1:]
  # return result


def create_node(op, name, inputs=[]):
  node = tf.NodeDef()
  node.op = op
  node.name = name
  for input_name in inputs:
    node.input.extend([input_name])
  return node


def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(
        tf.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass

def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(
        tf.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))
  except KeyError:
    pass

def set_attr_bool(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(b=value))
  except KeyError:
    pass


def set_attr_int(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(i=value))
  except KeyError:
    pass


def set_attr_float(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(f=value))
  except KeyError:
    pass


def set_attr_string(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(s=value))
  except KeyError:
    pass


def copy_attr(node, key, attr_value):
  try:
    node.attr[key].CopyFrom(attr_value)
  except KeyError:
    pass

def create_nodes_map(graph_def):
  """Builds a mapping of node names to their defs from the graph."""
  nodes_map = {}
  for node in graph_def.node:
    if node.name not in nodes_map.keys():
      nodes_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected.")
  return nodes_map


def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):[\d\*]+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name

def ensure_tensor_name_has_port(node_name):
  """Makes sure that a tensor name has :0 if no explicit port exists."""
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    name_with_port = node_name
  else:
    name_with_port = node_name + ":0"
  return name_with_port


def rename_node_inputs(input_graph_def, inputs_to_rename, output_nodes, preserve_names={}):
  output_graph_def = tf.GraphDef()
  canonical_inputs_to_rename = {}
  for key, value in inputs_to_rename.items():
    node_name = node_name_from_input(key)
    if node_name not in canonical_inputs_to_rename:
      canonical_inputs_to_rename[node_name] = []
    canonical_inputs_to_rename[node_name].append((key, value))

  for node in input_graph_def.node:
    if node.name in preserve_names:
      output_graph_def.node.extend([copy.deepcopy(node)])

    for index, input_full_name in enumerate(node.input):
      new_input_name = input_full_name
      already_visited = {}
      while node_name_from_input(new_input_name) in canonical_inputs_to_rename:
        input_node_name = node_name_from_input(new_input_name)
        if input_node_name in already_visited:
          raise ValueError("already visited")
        already_visited[input_node_name] = True

        any_match_found = False
        for source_name, dest_name in canonical_inputs_to_rename[input_node_name]:
          if source_name.endswith(":*"):
            is_match = True
            prefix, unused_name, suffix = node_name_parts_from_input(new_input_name)
            match_name = prefix + dest_name + suffix
          else:
            is_match = (ensure_tensor_name_has_port(source_name) == ensure_tensor_name_has_port(new_input_name))
            match_name = dest_name
          if is_match:
            new_input_name = match_name
            any_match_found = True
        if not any_match_found:
          break

      node.input[index] = new_input_name
    
    new_node = tf.NodeDef()
    new_node.CopyFrom(node)
    output_graph_def.node.extend([new_node])

  # extract subgraph to remove dead nodes
  return extract_sub_graph(output_graph_def, output_nodes)
  # return output_graph_def


def hash_node_def(node):
  # uint64 hash = Hash64String(node.op());
  # hash = Hash64Combine(hash, Hash64String(node.name()));
  # for (const string& input : node.input()) {
  #   hash = Hash64Combine(hash, Hash64String(CanonicalInputName(input)));
  # }
  # hash = Hash64Combine(hash, Hash64String(node.device()));
  # std::vector<string> attr_names;
  # attr_names.reserve(node.attr().size());
  # for (const auto& attr : node.attr()) {
  #   attr_names.push_back(attr.first);
  # }
  # std::sort(attr_names.begin(), attr_names.end());
  # string attr_serialized;
  # for (const string& attr_name : attr_names) {
  #   auto attr = node.attr().at(attr_name);
  #   attr.SerializeToString(&attr_serialized);
  #   hash = Hash64Combine(hash, Hash64String(attr_serialized));
  # }
  # return hash
  hash_obj = hashlib.md5(node.op.encode('utf-8'))
  hash_obj.update(node.name.encode('utf-8'))
  for input_name in node.input:
    input_name = ensure_tensor_name_has_port(input_name)
    hash_obj.update(input_name.encode('utf-8'))
  hash_obj.update(node.device.encode('utf-8'))

  attr_names = [attr for attr in node.attr]
  attr_names = sorted(attr_names)
  for attr_name in attr_names:
    attr = node.attr[attr_name]
    hash_obj.update(attr.SerializeToString())

  return hash_obj.digest()


def merge_adjacent_requantizes(input_graph_def, input_nodes, output_nodes):
  # TODO
  return input_graph_def


def merge_duplicate_nodes(input_graph_def, input_nodes, output_nodes):
  any_duplicates_found = True
  current_graph_def = input_graph_def
  while any_duplicates_found:
    any_duplicates_found = False
    # First arrange all of the nodes by a hash of their contents.
    hashed_nodes = {}
    for node in current_graph_def.node:
      nameless_node = tf.NodeDef()
      nameless_node.CopyFrom(node)
      if node.name not in input_nodes and node.name not in output_nodes:
        nameless_node.name = ""
      hash_value = hash_node_def(nameless_node)
      if hash_value not in hashed_nodes:
        hashed_nodes[hash_value] = []
      hashed_nodes[hash_value].append(node)
    
    # If we have multiple nodes with the same hash, then we know they're
    # duplicates and can be removed, unless they're stateful.
    inputs_to_rename = {}
    for hash_value, nodes in hashed_nodes.items():
      for i, node in enumerate(nodes):
        op_def = RegisteredOps[node.op]
        is_duplicate = i>0 and not op_def.is_stateful
        if is_duplicate:
          inputs_to_rename[node.name+":*"] = nodes[0].name
          any_duplicates_found = True
    if any_duplicates_found:
      current_graph_def = rename_node_inputs(
          current_graph_def, inputs_to_rename, output_nodes)
  return current_graph_def


def remove_redundant_quantizations(input_graph_def, input_nodes, output_nodes):
  nodes_map = create_nodes_map(input_graph_def)
  graph_outputs = {node_name_from_input(name) for name in output_nodes}
  inputs_to_rename = {}
  preserve_names = {}
  for node in input_graph_def.node:
    if node.op != "QuantizeV2":
      continue
    
    dequantize_node_name = node_name_from_input(node.input[0])
    if dequantize_node_name not in nodes_map:
      raise ValueError("Input node name '" + dequantize_node_name +
                         "' not found in node '" + node.name + "'")
    
    dequantize_node = nodes_map[dequantize_node_name]
    if dequantize_node.op != "Dequantize":
      continue
    if node.attr["T"] != dequantize_node.attr["T"]:
      continue

    min_node_name = node_name_from_input(node.input[1])
    max_node_name = node_name_from_input(node.input[2])
    min_node = nodes_map[min_node_name]
    max_node = nodes_map[max_node_name]

    if min_node.op != "Min" or max_node.op != "Max":
      print("Didn't find expected types on inputs : %s, %s." % (min_node.op,
                                                                max_node.op))
      continue
    
    for i in range(3):
      inputs_to_rename[node.name+":"+str(i)] = dequantize_node.input[i]
    
    # Are other sub-graphs using the float intermediate result? If so,
    # preserve it, but the input renaming still rewires the eight-bit ops
    # so they don't go through float.
    if dequantize_node_name in graph_outputs \
       or dequantize_node_name in output_nodes:
      preserve_names[dequantize_node_name] = True
  
  return rename_node_inputs(input_graph_def, inputs_to_rename, output_nodes, preserve_names)
  

def get_in_out_types(node_def):

  def add_arg_to_sig(arg_def, sig):
    original_size = len(sig)
    if arg_def.number_attr:
      repeats = node_def.attr[arg_def.number_attr]
      if repeats < 0:
        raise ValueError("Value for number_attr < 0")
      
      if arg_def.type_attr:
        dtype = node_def.attr[arg_def.type_attr].type
        for i in range(repeats):
          sig.append(dtype)
      elif arg_def.type != types_pb2.DT_INVALID:
        for i in range(repeats):
          sig.append(arg_def.type)
      else:
        raise ValueError("Missing type or type_attr field")
    elif arg_def.type_attr:
      dtype = node_def.attr[arg_def.type_attr].type
      sig.append(dtype)
    elif arg_def.type_list_attr:
      dtypes = node_def.attr[arg_def.type_list_attr]
      for dtype in dtypes:
        sig.append(dtype)
    elif arg_def.type != types_pb2.DT_INVALID:
      sig.append(arg_def.type)
    else:
      raise ValueError("No type fileds in argdef")
    # For all types that were added by this function call, make them refs.
    if arg_def.is_ref:
      for i in range(original_size, len(sig)):
        sig[i] = tf.Dtype(sig[i]).base_dtype.as_datatype_enum


  op_def = RegisteredOps[node_def.op]
  input_types = []
  for arg in op_def.input_arg:
    add_arg_to_sig(arg, input_types)
  output_types = []
  for arg in op_def.output_arg:
    add_arg_to_sig(arg, output_types)
  return input_types, output_types


def quantize_nodes(input_graph_def, input_nodes, output_nodes, ignore_names):
  output_graph_def = tf.GraphDef()
  # visied_nodes = {}
  for node in input_graph_def.node:
    if node.op not in QUANT_OP_SET \
       or should_ignore(ignore_names, node.name) \
       or "attn_score_softmax" in node.name: # quantize on this op will cause error
       # rm_redundant_quantization causes errors on QuantizedAdd/Mul and Requantize
       # maybe because non-positive scalar results in equal min and max
      new_node = tf.NodeDef()
      new_node.CopyFrom(node)
      output_graph_def.node.extend([new_node])
      continue

    op_info = QUANT_OP_SET[node.op]
    input_types, output_types = get_in_out_types(node)
    # # skip if not a float op
    all_float = True
    for i, iname in enumerate(node.input):
      if i in op_info["unquantized_inputs"]:
        continue
      if iname.startswith("^"):
        all_float = False
        break
      if input_types[i] != DT_FLOAT.as_datatype_enum:
        all_float = False
    for otype in output_types:
      if otype != DT_FLOAT.as_datatype_enum:
        all_float = False
        
    if not all_float:
      new_node = tf.NodeDef()
      new_node.CopyFrom(node)
      output_graph_def.node.extend([new_node])
      continue

    namespace_prefix = node.name + "_eightbit"

    # if "memory_attention/mul_1" in node.name:
    #   print(node)
    #   exit()

    # quantize all inputs
    quantized_input_names = []
    for i, input_name in enumerate(node.input):
      # skip non-float input
      if i in op_info["unquantized_inputs"]:
        continue
      
      unique_input_name = namespace_prefix + "/" + unique_node_name_from_input(input_name)

      assert not input_name.startswith("^")

      control_input_name = node_name_from_input(input_name)

      reshape_dims = create_constant_node(
          unique_input_name+"/reshape_dims", -1, DT_INT32, [1])
      reshape_dims.input.extend(["^" + control_input_name])
      output_graph_def.node.extend([reshape_dims])

      reduction_dims = create_constant_node(
          unique_input_name+"/reduction_dims", 0, DT_INT32, [1])
      reduction_dims.input.extend(["^" + control_input_name])
      output_graph_def.node.extend([reduction_dims])

      reshape_node = create_node(
          "Reshape", unique_input_name+"/reshape",
          [input_name, reshape_dims.name])
      set_attr_dtype(reshape_node, "T", DT_FLOAT)
      output_graph_def.node.extend([reshape_node])
      
      min_node = create_node(
          "Min", unique_input_name+"/min", 
          [reshape_node.name, reduction_dims.name])
      set_attr_dtype(min_node, "T", DT_FLOAT)
      set_attr_bool(min_node, "keep_dims", False)
      output_graph_def.node.extend([min_node])

      max_node = create_node(
          "Max", unique_input_name+"/max", 
          [reshape_node.name, reduction_dims.name])
      set_attr_dtype(max_node, "T", DT_FLOAT)
      set_attr_bool(max_node, "keep_dims", False)
      output_graph_def.node.extend([max_node])

      quantize_node = create_node(
          "QuantizeV2", unique_input_name+"/quantize", 
          [input_name, min_node.name, max_node.name])
      set_attr_dtype(quantize_node, "T", DT_QUINT8)
      set_attr_string(quantize_node, "mode", QUANT_MODE)
      output_graph_def.node.extend([quantize_node])
      quantized_input_names.append(quantize_node.name)
      # visited_nodes[input_name] = quantize_node.name
    # end for
  
    # Quantized version of the current op
    quantized_main_node = create_node("Quantized"+node.op, node.name+"/eightbit")
    for attr in op_info["attrs_to_copy"]:
      copy_attr(quantized_main_node, attr, node.attr[attr])
    for dtype, value in op_info["dtypes_to_set"].items():
      set_attr_dtype(quantized_main_node, dtype, value)
    
    quantized_input_index = 0
    for i, input_name in enumerate(node.input):
      if i in op_info["unquantized_inputs"]:
        quantized_main_node.input.extend([input_name])
      else:
        quantized_input_name = quantized_input_names[quantized_input_index]
        quantized_main_node.input.extend([quantized_input_name + ":0"])
        quantized_input_index += 1
    if op_info["min_max_order"] == CONTIGUOUS_MIN_MAX:
      for quantized_input_name in quantized_input_names:
        quantized_main_node.input.extend([quantized_input_name + ":1"])
        quantized_main_node.input.extend([quantized_input_name + ":2"])
    else:
      for quantized_input_name in quantized_input_names:
        quantized_main_node.input.extend([quantized_input_name + ":1"])
      for quantized_input_name in quantized_input_names:
        quantized_main_node.input.extend([quantized_input_name + ":2"])
    output_graph_def.node.extend([quantized_main_node])

    eight_bit_node_name = quantized_main_node.name
    if op_info["output_bit_depth"] == DT_QINT32:
      quantized_main_outputs = []
      for i in range(3):
        quantized_main_outputs.append(quantized_main_node.name+":"+str(i))
      # shrink output down from 32 bits to 8
      # dynamically measure the range each time
      # cound cause max<min error, verified by a prefined replacement
      # TODO uncomment
      requant_range_node = create_node(
          "RequantizationRange", 
          quantized_main_node.name+"/requant_range")
      set_attr_dtype(requant_range_node, "Tinput", DT_QINT32)
      requant_range_node.input.extend(quantized_main_outputs)
      output_graph_def.node.extend([requant_range_node])

      requant_min_input = requant_range_node.name + ":0"
      requant_max_input = requant_range_node.name + ":1"

      # predefined range for testing
      # requant_min_node = create_constant_node(quantized_main_node.name+"/requant_range_min", 0., DT_FLOAT)
      # requant_min_node.input.extend(["^" + quantized_main_node.name])
      # requant_max_node = create_constant_node(quantized_main_node.name+"/requant_range_max", 100., DT_FLOAT)
      # requant_max_node.input.extend(["^" + quantized_main_node.name])
      # requant_min_input = requant_min_node.name
      # requant_max_input = requant_max_node.name
      # output_graph_def.node.extend([requant_min_node, requant_max_node])

      requantize_node = create_node(
          "Requantize", quantized_main_node.name+"/requantize",
          quantized_main_outputs + [requant_min_input,
                                    requant_max_input])
      set_attr_dtype(requantize_node, "Tinput", DT_QINT32)
      set_attr_dtype(requantize_node, "out_type", DT_QUINT8)
      output_graph_def.node.extend([requantize_node])
      eight_bit_node_name = requantize_node.name
    
    # convert the 8-bit result back into float for the final output
    dequantize_node = create_node(
        "Dequantize", node.name, 
        [eight_bit_node_name + ":0",
        eight_bit_node_name + ":1",
        eight_bit_node_name + ":2"])
    set_attr_dtype(dequantize_node, "T", DT_QUINT8)
    set_attr_string(dequantize_node, "mode", QUANT_MODE)
    output_graph_def.node.extend([dequantize_node])
  optlib.ensure_graph_is_valid(output_graph_def)

  # If we've ended up with two Requantize ops in a row (for example if there
  # was a Conv2D feeding into a FakeQuantWithMinMaxVars) merge them together,
  # using the trained range from the second op.
  # output_graph_def = merge_adjacent_requantizes(output_graph_def, input_nodes, output_nodes)

  # There can be duplicate quantize nodes if multiple ops pull from a single
  # input, which makes it harder to remove redundant ones, so strip them out.
  output_graph_def = merge_duplicate_nodes(output_graph_def, input_nodes, output_nodes)
  optlib.ensure_graph_is_valid(output_graph_def)
  # Look for Dequantizes that immediately go into Quantizes, and remove them
  # since the two together cancel each other out. This allows us to keep the
  # data flow in eight bit where two adjacent ops are in eight bit, but still
  # keep interoperability with float ops.
  output_graph_def = remove_redundant_quantizations(output_graph_def, input_nodes, output_nodes)
  optlib.ensure_graph_is_valid(output_graph_def)
  return output_graph_def
