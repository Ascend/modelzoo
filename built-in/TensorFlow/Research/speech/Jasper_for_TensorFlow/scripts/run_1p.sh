#!/bin/bash


export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export SLOG_PRINT_TO_STDOUT=0

device_group="0"

mkdir -p ${currentDir}/result/1p > /dev/null 2>&1
for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train_1p.sh ${device_phy_id} & "
    ${currentDir}/train_1p.sh ${device_phy_id}
done


echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train_1p.sh exit "

