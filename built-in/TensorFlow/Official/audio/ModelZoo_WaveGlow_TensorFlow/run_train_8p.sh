# -*- coding: utf-8 -*-


bash ./run_npu_env.sh
source ./run_npu_env.sh

export JOB_ID=80000

export RANK_SIZE=8
export RANK_TABLE_FILE=./8p.json


sleep 5

device_group='0 1 2 3 4 5 6 7'
for deviceid in ${device_group}
do
    export  DEVICE_ID=${deviceid}
    export RANK_ID=${deviceid}
    python3  train.py  --learning_rate 6e-4  --epochs 110  --decay_steps  2000  --batch_size  12  &
done

