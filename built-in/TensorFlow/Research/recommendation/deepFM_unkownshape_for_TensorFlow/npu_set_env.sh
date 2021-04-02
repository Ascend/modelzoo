# main env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/code:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/atc/python/site-packages
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=600
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export EXPERIMENTAL_DYNAMIC_PARTITION=1
#export HYBRID_PROFILING_LEVEL=1
#export ENABLE_FORCE_V2_CONTROL=1


# profiling env
export PROFILING_MODE={PROFILING_MODE}
export PROFILING_OPTIONS=\{\"output\":\"/data/h00562660/profiling/\"\,\"training_trace\":\"{training_trace_flag}\"\,\"task_trace\":\"{task_trace_flag}\"\,\"aicpu\":\"{aicpu_trace_flag}\"\,\"fp_point\":\"{FP_POINT}\"\,\"bp_point\":\"{BP_POINT}\"\}

# debug env
export ASCEND_GLOBAL_LOG_LEVEL=3
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1
#export TE_PARALLEL_COMPILER=0

# system env
ulimit -c unlimited
