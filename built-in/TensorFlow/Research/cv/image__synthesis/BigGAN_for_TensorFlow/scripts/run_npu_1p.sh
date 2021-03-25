#!/bin/bash


rm -rf /var/log/npu/slog/host-0/*

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
source ${CURRENT_DIR}/env.sh

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p
export RANK_TABLE_FILE=${CURRENT_DIR}/1p.json

export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
device_group="0"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
	${CURRENT_DIR}/train_1p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log
