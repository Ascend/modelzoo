#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
dir=`pwd`

#cp -rf ./config /tmp/
export JOB_ID=10086
#export PROFILING_DIR=/var/log/npu/profiling/container/0
export DEVICE_ID=0
#export PROFILING_MODE=true
export PRINT_MODEL=1
#export ENABLE_DATA_PRE_PROC=1
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=/home/lxh/config/new_rank_table_1p.json
export FUSION_TENSOR_SIZE=1000000000
export PYTHONPATH=${dir}
export LD_LIBRARY_PATH=/usr/local/HiAI/runtime/lib64/
/usr/local/HiAI/runtime/bin/TdtMain --configfile=/home/lxh/test/config/job_tdt_2p_$DEVICE_ID.json  &
sleep 5

python3.6 res50.py --config_file res50_baseline 
