#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/python3.7.5/lib/python3.7.5/site-packages/tensorflow_core/:/usr/local/python3.7.5/lib/python3.7.5/site-packages/tensorflow_core/python/:/usr/local/Ascend/opp/:/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/toolkit/lib64:/usr/local/Ascend/atc/lib64
#export TF_CPP_MIN_VLOG_LEVEL=0
#export GE_USE_STATIC_MEMORY=1
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin:/usr/local/Ascend/toolkit/bin:/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
#export NEW_GE_FE_ID=1
#export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/ops/framework/built-in/tensorflow
export PYTHONPATH=/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg:/usr/local/Ascend/atc/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages
export WHICH_OP=GEOP
export SOC_VERSION=Ascend910

currentDir=$(cd "$(dirname "$0")"; pwd)
# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=${currentDir}/8p.json
export RANK_SIZE=8
export RANK_ID=npu8p
export SLOG_PRINT_TO_STDOUT=0
export HCCL_CONNECT_TIMEOUT=600

# dump graph
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=1
export PRINT_MODEL=1

device_group="0 1 2 3 4 5 6 7"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
    ${currentDir}/train_tacotron_8p.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log

