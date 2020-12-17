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
#!/bin/bash
source /root/archiconda3/bin/activate ci3.7
#source /ssd/ssd0/mindconda3/bin/activate
export RUNTIME_MODE="rpc_cloud"
export SLOG_PRINT_TO_STDOUT=1
export ME_DRAW_GRAPH=1
export GE_TRAIN=1
export HCCL_FLAG=1
export DEPLOY_MODE=0
export AICPU_FLAG=1
#export REUSE_MEMORY=1
unset REUSE_MEMORY
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
#export GLOG_v=1

# use for debug
export DUMP_GE_GRAPH=1
#unset DUMP_GE_GRAPH
#export DUMP_OP=1
unset DUMP_OP
#export DISABLE_REUSE_MEMORY=1

export LD_LIBRARY_PATH=/root/archiconda3/envs/ci3.7/lib/:/usr/local/HiAI/runtime/lib64:/usr/local/HiAI/add-ons
export ME_TBE_PLUGIN_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe
export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:$PATH

export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/libhccl.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so
export FE_FLAG=1
export OPTION_PROTO_LIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so
export OP_PROTOLIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so

# !!! ADD YOUR OWN PYTHONPATH
export PYTHONPATH=/PATH/TO/CONFIGED/ME_CenterFace_To_modelzoo/:${PYTHONPATH}
export GE_USE_STATIC_MEMORY=1
