#!/bin/bash


rm -rf /var/log/npu/slog/host-0/*

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
#source ${CURRENT_DIR}/env.sh
 
# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p


device_group="0"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] start: train.sh ${device_phy_id} & " >> main.log
    ${CURRENT_DIR}/train_wideanddeep_criteo_1p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] all train.sh exit " >> main.log

