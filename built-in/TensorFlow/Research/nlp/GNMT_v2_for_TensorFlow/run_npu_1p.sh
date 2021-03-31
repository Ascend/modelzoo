#!/bin/bash
 
# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=0
export RANK_TABLE_FILE=1p.json
source env.sh

bash train_gnmt_1p.sh 1 &


echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

