#!/bin/bash
# set env

export HCCL_CONNECT_TIMEOUT=600

export TF_CPP_MIN_LOG_LEVEL=3

currentDir=$(cd "$(dirname "$0")"; pwd)
rm -rf /var/log/npu/slog
rm -f *.txt
rm -f *.pbtxt
rm -fr dump*
rm -f  *.log

# user env
export JOB_ID=9999001
export RANK_SIZE=8
export RANK_TABLE_FILE=${currentDir}/../config/rank_table_unet_8p.json

device_group="0 1 2 3 4 5 6 7"
RES=${1}
mkdir -p ${RES}
for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${RES}/main.log
    ${currentDir}/train_8p.sh $2 $3 ${device_phy_id} ${1} &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${RES}/main.log
