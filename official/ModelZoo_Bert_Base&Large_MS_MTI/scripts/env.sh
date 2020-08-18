#!/bin/bash
export RUNTIME_PATH=/usr/local/Ascend
unset WHICH_OP
unset NEW_GE_FE_ID
unset OPTION_EXEC_EXTERN_PLUGIN_PATH
unset PYTHONPATH
unset ME_DRAW_GRAPH
unset DUMP_GE_GRAPH
unset DISABLE_REUSE_MEMORY
unset DUMP_OP
export GLOG_v=1
export SLOG_PRINT_TO_STDOUT=1
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:/usr/local/Ascend/fwkacllib/lib64/:$LD_LIBRARY_PATH
export ME_TBE_PLUGIN_PATH=/usr/local/Ascend/opp/framework/built-in/tensorflow
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/librts_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/librts_engine.so
export PYTHONPATH=/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/
export PATH=/usr/local/Ascend/fwkacllib/ccec_compiler/bin:$PATH
export FE_FLAG=1
export AICPU_FLAG=1
export OPTION_PROTO_LIB_PATH=/usr/local/Ascend/opp/op_proto/built-in/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
export MINDSPORE_CONFIG_PATH=$1/config/
export ME_DRAW_GRAPH=1
export DUMP_GE_GRAPH=4

unset OP_NO_REUSE_MEM_BY_TYPE
unset OP_NO_REUSE_MEM_BY_NAME

export P_NUM=32
export MSLIBS_SERVER=10.29.74.101
export DEVICE_ID=0
