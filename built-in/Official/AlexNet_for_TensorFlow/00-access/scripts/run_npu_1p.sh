#!/bin/bash

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
 
# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p
export SLOG_PRINT_TO_STDOUT=0


device_group="0"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
    ${CURRENT_DIR}/train_alexnet_1p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

