#!/bin/bash

mkdir -p result/1p

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_SIZE=1

export SLOG_PRINT_TO_STDOUT=0

device_group=0

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${currentDir}/result/1p/main.log
    ${currentDir}/train_1p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${currentDir}/result/1p/main.log

