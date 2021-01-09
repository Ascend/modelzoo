#!/bin/bash

# set env
export HCCL_CONNECT_TIMEOUT=600

currentDir=$(cd "$(dirname "$0")"; pwd)
echo "=========>>> Current dir: "${currentDir}

resultDir=/cache/result
mkdir -p ${resultDir}

# user env
export JOB_ID=9999001
export RANK_SIZE=1

export SLOG_PRINT_TO_STDOUT=0

device_group=${DEVICE_ID}

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${resultDir}/main.log
    bash ${currentDir}/train_1p.sh ${device_phy_id} ${currentDir} ${resultDir} &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${resultDir}/main.log

