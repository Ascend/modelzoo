#!/bin/bash

#main
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/python3/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/usr/local/Ascend/fwkacllib/ops/op_impl/built-in/ai_core/tbe/:/usr/local/Ascend/fwkacllib/python/site-packages/te:/usr/local/Ascend/fwkacllib/python/site-packages/topi:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH

export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp

export SOC_VERSION=Ascend910
export DDK_VERSION_PATH=/usr/local/Ascend/fwkacllib/ddk_info

#debug
#export PRINT_MODEL=1
#export DUMP_OP=0
#export DUMP_GE_GRAPH=3
#export DUMP_GRAPH_LEVEL=3

export EXPERIMENTAL_DYNAMIC_PARTITION=1

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error


#profiling
export PROFILING_MODE=false  # true if need profiling
export PROFILING_OPTIONS="{\"output\":\"/autotest/profiling\",\"task_trace\":\"on\",\"training_trace\":\"on\",\"aicpu\":\"on\",\"fp_point\":\"\",\"bp_point\":\"\",\"aic_metrics\":\"PipeUtilization\"}"

