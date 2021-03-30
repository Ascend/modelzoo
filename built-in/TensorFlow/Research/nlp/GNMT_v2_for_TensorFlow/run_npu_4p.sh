#!/bin/bash

# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=4p.json
export RANK_SIZE=4
source env.sh

device_group="4 5 6 7"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
    ./train_gnmt_4p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

