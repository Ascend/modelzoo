#!/bin/bash


rm -rf /var/log/npu/slog/host-0/*

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
source ${CURRENT_DIR}/env.sh
 
# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p
export RANK_TABLE_FILE=${CURRENT_DIR}/1p.json

#export DUMP_OP=1
#export DISABLE_REUSE_MEMORY=1
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1




export PROFILING_MODE=true
export AICPU_PROFILING_MODE=true
export PROFILING_OPTIONS=training_trace:task_trace
export PROFILING_DIR=/var/log/npu/profiling/container/0
export FP_POINT=megatron/GatherV2_1
export BP_POINT=loss_scale/gradients_4/loss_scale/megatron/gpt2_block_0/TransformerLayer/layer_normalization/sub_grad/Sum




device_group="0"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
    ${CURRENT_DIR}/train_1p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

