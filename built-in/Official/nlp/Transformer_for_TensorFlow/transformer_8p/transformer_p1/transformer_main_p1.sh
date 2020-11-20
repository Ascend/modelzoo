#!/bin/bash
dir=`pwd`
export JOB_ID=10086
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=8
export RANK_TABLE_FILE=${dir}/new_rank_table_8p.json

export GE_USE_STATIC_MEMORY=1
cd ../../

sh train-ende.sh
