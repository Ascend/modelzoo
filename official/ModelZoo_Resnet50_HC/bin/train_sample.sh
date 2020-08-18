#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

env > ${currentDir}/env_${device_id}.log

#mkdir exec path
mkdir -p ${currentDir}/${device_id}
rm -rf ${currentDir}/${device_id}/*
cd ${currentDir}/${device_id}

#start exec
python3.7 {RUN_ALGORITHM_CMD} {CHECKPOINT_DIR} > ${currentDir}/train_${device_id}.log 2>&1
if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
fi
