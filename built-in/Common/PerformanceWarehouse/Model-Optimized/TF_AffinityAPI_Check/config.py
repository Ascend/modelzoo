#! -*- coding:utf-8 -*-
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
# ============================================================================


# TFAdpater
DATASET_OP = ["MakeIterator", "IteratorV2", "DatasetDataset"]
# Dataset接口,判断是否具备可下沉条件
DATASET_INTERFACE_SUGGESTIONS = "网络没有使用Dataset队列模式, 请尝试修改为Dataset接口模式"
GETNEXT_API = ["make_one_shot_iterator"]
# 判断是否下沉
GETNEXT_SUGGESTIONS = "Getnext算子没有下沉,请检查脚本预处理队列写法, 尝试make_one_shot_iterator替换为make_initializable_iterator."

# 精度模式非allow_mix_precision
PRECISION_MODE_SUGGESTIONS = "精度模式是{}, 建议查看迁移指南设置precision_mode为allow_mix_precision."
# mix_compile_mode非False
MIX_COMPILE_MODE_SUGGESTIONS = "混合计算(mix_compile_mode)为True, 建议设置为False."
# op_debug_level非0
OP_DEBUG_LEVEL = "维测接口(op_debug_level)值为1, 建议设置为0."

# profiling
DROPOUT_NODE = "dropout/random_uniform/RandomUniform"
DROPOUT_SUGGESTIONS = "网络中使用的dropout/random_uniform/RandomUniform是AICPU算子, 建议使用华为亲和npu_ops.dropout接口进行替换."

DYNAMIC_RNN_NODE = ["StackPushV2", "StackPopV2"]
DYNAMIC_RNN_APIS= ["tf.nn.dynamic_rnn", "tf.nn.bidirectional_dynamic_rnn", "tf.nn.static_rnn",
                    "tf.contrib.rnn.stack_bidirectional_dynamic_rnn", "tf.nn.static_bidirectional_rnn"]
DYNAMIC_RNN_SUGGESTIONS = "网络使用了大量的StackPushV2、StackPopV2 AICPU算子, 建议把TF原生接口替换为华为提供的亲和接口大kernel接口 " \
                          "estimator.npu.npu_dynamic_rnn DynamicRNN或者DynamicGruV接口."

AICPU_SUGGESTIONS = "网络中存在数据类型是INT64的AICPU算子, 请查看log或profiling数据, 尝试修改为INT32走AICORE."


TRANSPOSED_SUGGESTIONS = "网络中存在一些耗时较长的TransposeD算子, 请根据profiling文件设置Transpose白名单."

# training code
LSTM_APIS = ["tf.nn.rnn_cell.BasicLSTMCell", "tf.contrib.rnn.LayerNormBasicLSTMCell",
             "tf.contrib.rnn.BasicLSTMCell"]
LSTM_SUGGESTIONS = "网络中存在LSTM接口, 建议替换为华为亲和接口npu_ops.basic_lstm_cell."

FINITE_APIS = ["tf.is_finite", "tf.debugging.is_finite", "tf.math.is_finite",
               "tf.compat.v1.debugging.is_finite", "tf.compat.v1.is_finite", "tf.compat.v1.math.is_finite"]
FINITE_SUGGESTIONS = "网络中使用了溢出检测API, 可删除对应API, 使用华为loss scale, 自动有溢出检测功能."

# npu log
DATASET_LOG_SUGGESTIONS = "Dataset预处理存在性能瓶颈，请分析训练脚本."




