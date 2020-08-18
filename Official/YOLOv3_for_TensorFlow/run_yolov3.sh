#!/bin/bash
rm -rf Onnxgraph
rm -rf Partition
rm -rf OptimizeSubGraph
rm -rf Aicpu_Optimized
rm *txt
rm -rf result_$RANK_ID



export RANK_ID=$1
export RANK_SIZE=$2
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export RANK_TABLE_FILE=rank_table.json
export JOB_ID=123678
export FUSION_TENSOR_SIZE=1000000000

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

#sleep 5
taskset -c  $PID_START-$PID_END python3 $3/train.py \
--mode $4

mkdir graph
mv *.txt graph
mv *.pbtxt graph
