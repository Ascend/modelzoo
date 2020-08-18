# main env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID={JOB_ID}
export RANK_TABLE_FILE={RANK_TABLE_FILE}
export RANK_SIZE={RANK_SIZE}
export RANK_INDEX={RANK_INDEX}
export RANK_ID={RANK_ID}

# profiling env
export PROFILING_MODE={PROFILING_MODE}
export AICPU_PROFILING_MODE={AICPU_PROFILING_MODE}
export PROFILING_OPTIONS={PROFILING_OPTIONS}
export FP_POINT={FP_POINT}
export BP_POINT={BP_POINT}

# debug env
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=2
#export DUMP_OP=1
#export DUMP_OP_LESS=1
#export PRINT_MODEL=1
#export TE_PARALLEL_COMPILER=0

#eventID 1024规避
#export AICPU_CONSTANT_FOLDING_ON=1

#export OFF_CONV_CONCAT=1
#export OFF_CONV_CONCAT_SPLIT=1

#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=1
