# -*- coding: utf-8 -*-


bash ./run_npu_env.sh
source ./run_npu_env.sh

export JOB_ID=80000

export RANK_SIZE=1
#export RANK_TABLE_FILE=./1p.json

sleep 5

deviceid=0
export  DEVICE_ID=${deviceid}
export RANK_ID=${deviceid}
python3  train.py  --learning_rate 6e-4  --epochs 110  --decay_steps  10000  --batch_size  12  


