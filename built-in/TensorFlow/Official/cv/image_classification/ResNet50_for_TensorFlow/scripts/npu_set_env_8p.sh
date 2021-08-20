# main env
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=resnet50-1p
export RANK_TABLE_FILE=${currentDir}/../src/configs/8p.json
export RANK_SIZE=8
export RANK_INDEX=0
export RANK_ID=0

# profiling env
export PROFILING_MODE=false
export AICPU_PROFILING_MODE=false
export PROFILING_OPTIONS=task_trace:training_trace
export FP_POINT=fp32_vars/conv2d/Conv2Dfp32_vars/BatchNorm/FusedBatchNormV3_Reduce
export BP_POINT=loss_scale/gradients/AddN_70/FusedMulAddNL2loss

# debug env
#export DUMP_GE_GRAPH=2
#export DUMP_OP=1
#export DUMP_OP_LESS=1
#export PRINT_MODEL=1
#export TE_PARALLEL_COMPILER=0

# system env
ulimit -c unlimited
