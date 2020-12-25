#!/bin/bash

export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600
export DUMP_GE_GRAPH=2
export PRINT_MODEL=1

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export SLOG_PRINT_TO_STDOUT=0

device_group="0"

logDir=${currentDir}/result/8p
[ -d "${currentDir}/result/1p" ] && logDir=${currentDir}/result/1p
for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: eval_1p.sh ${device_phy_id} & "
    bash ${currentDir}/eval_1p.sh ${device_phy_id}
done

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all eval_1p.sh exit "

