# main env
#export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=123456702
#请根据hccl.json文件实际路径进行配置
export RANK_TABLE_FILE=/network/ResNext50_for_TensorFlow/hccl.json
export RANK_SIZE=$2
export RANK_ID=$1
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID

# profiling env
export PROFILING_MODE=FALSE
export AICPU_PROFILING_MODE=TRUE
export PROFILING_OPTIONS=training_trace:task_trace
export FP_POINT=fp32_vars/conv2d/Conv2Dfp32_vars/BatchNorm/FusedBatchNormV3_Reduce
export BP_POINT=loss_scale/gradients/fp32_vars/BatchNorm/FusedBatchNormV3_grad/FusedBatchNormGradV3_Reduce

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
