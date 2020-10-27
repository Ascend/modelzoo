#!/bin/bash

#if [ ! -d "result/8p" ]; then
mkdir -p result/8p
#else
#    rm -rf result/8p/*
#fi

# set env
#################### configure the env ####################
#
#
#
#
###########################################################

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=${currentDir}/8p.json
export RANK_SIZE=8
export RANK_ID=ascend8p
export SLOG_PRINT_TO_STDOUT=0
export HCCL_CONNECT_TIMEOUT=600

device_group="0 1 2 3 4 5 6 7"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${currentDir}/result/8p/main.log
    ${currentDir}/train_8p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${currentDir}/result/8p/main.log

