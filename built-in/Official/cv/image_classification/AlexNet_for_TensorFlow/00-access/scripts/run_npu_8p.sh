#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)
# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=${currentDir}/8p.json
export RANK_SIZE=8
export RANK_ID=npu8p
export SLOG_PRINT_TO_STDOUT=0
export HCCL_CONNECT_TIMEOUT=600

device_group="0 1 2 3 4 5 6 7"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
    ${currentDir}/train_alexnet_8p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

