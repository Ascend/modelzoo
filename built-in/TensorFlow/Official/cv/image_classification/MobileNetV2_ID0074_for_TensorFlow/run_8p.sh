#!/bin/bash

mkdir -p result/8p

# set env
export HCCL_CONNECT_TIMEOUT=600

currentDir=$(cd "$(dirname "$0")"; pwd)

# DataDump
export FLAG_ENABLE_DUMP=False
export DUMP_PATH=/var/log/npu/dump
export DUMP_STEP="0|2"
export DUMP_MODE="all"
mkdir -p $DUMP_PATH
chown -R HwHiAiUser:HwHiAiUser $DUMP_PATH

# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=${currentDir}/8p.json
export RANK_SIZE=8
export RANK_ID=ascend8p

export SLOG_PRINT_TO_STDOUT=0

device_group="0 1 2 3 4 5 6 7"

num_cpus=$(getconf _NPROCESSORS_ONLN)
num_cpus_per_device=$((num_cpus/8))

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${currentDir}/result/8p/main.log
    start_id=$((num_cpus_per_device*device_phy_id))
    end_id=$((num_cpus_per_device*device_phy_id+num_cpus_per_device-1))
    taskset -c ${start_id}-${end_id} ${currentDir}/train_8p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${currentDir}/result/8p/main.log

