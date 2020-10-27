#!/bin/bash
dir=`pwd`
export JOB_ID=10086
export DEVICE_ID=2
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=${dir}/new_rank_table_1p.json
cd ../
sh train-ende.sh