# coding=utf-8
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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The tools to inference onnx model in cpu to get the benchmark output."""
import onnxruntime
import numpy as np
from data_convert import binary2np


def inference_onnx_cpu(onnx_file, input_data_path, input_shape, input_dtype):
    """Inference of the onnx in cpu to get the benchmark output.

    :param onnx_file: the onnx file
    :type onnx_file: str
    :param input_data_path: input data file, the .bin or .data format
    :type input_data_path: str
    :param input_shape: the shape of the input
    :type input_shape: list
    :param input_dtype: the dtype of input
    :type input_dtype: str
    """
    input_data = binary2np(input_data_path, input_shape, input_dtype)
    sess = onnxruntime.InferenceSession(onnx_file)
    output_nodes = sess.get_outputs()[0].name
    input_nodes = sess.get_inputs()[0].name
    res = sess.run([output_nodes], {input_nodes: input_data})
    res.tofile("expect_out.data")
