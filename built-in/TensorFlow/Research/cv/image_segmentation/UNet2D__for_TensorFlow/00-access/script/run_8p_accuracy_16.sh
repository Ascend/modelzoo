#!/bin/bash

# set env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export HCCL_CONNECT_TIMEOUT=300

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

# user env
export JOB_ID=NPU20210126
export RANK_SIZE=8
export RANK_TABLE_FILE=${currentDir}/../npu_config/8p.json

# debug env
export ASCEND_GLOBAL_LOG_LEVEL=3

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

fold=$1
device_group="0 1 2 3 4 5 6 7"

if [ x"${fold}" = x"all" ] ;
then
    for((i=0;i<=4;i++));
    do
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold${i} train start"
        for device_index in ${device_group}
        do
            RANK_ID=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/exec_8p_16.sh ${i} &
        done

        wait
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold${i} train end"
    done

    python3.7 ${currentDir}/../utils/parse_results.py --model_dir ${currentDir}/../results --exec_mode convergence --env TF-AMP_8GPU
else
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train start"
    for device_index in ${device_group}
    do
        RANK_ID=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/exec_8p_16.sh ${fold} &
    done

    wait
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train end"
fi

#python3 main.py --data_dir ./data --model_dir ./results --batch_size 1 --exec_mode train --crossvalidation_idx 1 --xla --amp
