#!/bin/bash

# set env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export HCCL_CONNECT_TIMEOUT=300

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

upDir=$(dirname "$PWD")
out_dir=${upDir}/output

# user env
export JOB_ID=NPU20210126
export RANK_SIZE=8
export RANK_TABLE_FILE=${currentDir}/8p.json

# debug env
export ASCEND_GLOBAL_LOG_LEVEL=3
#export SLOG_PRINT_TO_STDOUT=0
#export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVEL=3
#export PROFILING_MODE=true
#export PROFILING_OPTIONS='{"output":"/npu/y00535628/profiling","training_trace":"on","task_trace":"off","aicpu":"off","fp_point":"conv3d/Conv3D","bp_point":"gradients/conv3d_22/BiasAdd_grad/BiasAddGrad"}'


data_dir=$1
fold=$2

device_group="0 1 2 3 4 5 6 7"

if [ x"${fold}" = x"all" ] ;
then
    for((i=0;i<=4;i++));
    do
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold${i} train start"
        for device_index in ${device_group}
        do
            RANK_ID=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/train_performance_8p.sh ${data_dir} ${i} &
        done

        wait
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold${i} train end"
    done
else
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train start"
    for device_index in ${device_group}
    do
        RANK_ID=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/train_performance_8p.sh ${data_dir} ${fold} &
    done

    wait
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train end"
fi

