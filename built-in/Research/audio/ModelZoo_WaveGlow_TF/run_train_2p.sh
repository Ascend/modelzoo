# -*- coding: utf-8 -*-


bash ./run_npu_env.sh
source ./run_npu_env.sh

export JOB_ID=80000

export RANK_SIZE=2
export RANK_TABLE_FILE=./2p.json


sleep 5

device_group='0 1'
for deviceid in ${device_group}
do
    export  DEVICE_ID=${deviceid}
    export RANK_ID=${deviceid}
    python3  train.py  --learning_rate 6e-4  --epochs 100  --decay_steps  8000  --batch_size  12  &
done

