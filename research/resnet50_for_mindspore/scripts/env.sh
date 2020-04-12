#!/bin/bash
source /root/miniconda3/bin/activate ci3.7
export RUNTIME_MODE="rpc_cloud"
export SLOG_PRINT_TO_STDOUT=1
export ME_DRAW_GRAPH=1
export GE_TRAIN=1
export HCCL_FLAG=1
export DEPLOY_MODE=0
export REUSE_MEMORY=1
export DUMP_GE_GRAPH=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

export LD_LIBRARY_PATH=/opt/npu/miniconda3/envs/ci3.7/lib/:/usr/local/HiAI/runtime/lib64
export ME_TBE_PLUGIN_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe
export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:$PATH
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so

export FE_FLAG=1
export AICPU_FLAG=1
export OPTION_PROTO_LIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so
export OP_PROTOLIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so

export PYTHONPATH=/opt/npu/tlw/script/modelzoo_resnet50:${PYTHONPATH}
