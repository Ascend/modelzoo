#!/bin/bash
export JOB_ID=10086
export FUSION_TENSOR_SIZE=1000000000
export EXPERIMENTAL_DYNAMIC_PARTITION=1

#dump
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1

#profiling
#export PROFILING_OPTIONS=\{\"output\":\"/home/w00348617/GNMT_chujun/profiling\"\,\"training_trace\":\"on\"\,\"task_trace\":\"on\"\,\"aicpu\":\"on\"\,\"fp_point\":\"{FP_POINT}\"\,\"bp_point\":\"{BP_POINT}\"\,\"aic_metrics\":\"PipeUtilization\"\}

#env
export GE_TRAIN=0
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/Ascend/tfplugin
export PATH=/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/python3.7.5/bin/sacrebleu/:$PATH
export PYTHONPATH=$PYTHONPATH:/usr/local/python3.7.5/bin/sacrebleu/
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/runtime/lib64/tbe_plugin/bert
export WHICH_OP=GEOP
export DDK_VERSION_FLAG=1.60.T17.B830
export NEW_GE_FE_ID=1
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/librts_engine.so
export OP_PROTOLIB_PATH=/usr/local/Ascend/runtime/ops/op_proto/built-in
export ASCEND_OPP_PATH=/usr/local/Ascend/ops

#log
export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
