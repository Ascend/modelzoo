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

#mkdir exec path
mkdir -p ${currentDir}/result/1p/${device_id}
rm -rf ${currentDir}/result/1p/${device_id}/*
cd ${currentDir}/result/1p/${device_id}

#start exec
python3.7 ${currentDir}/train.py --config_file vgg16_config_1p > ${currentDir}/result/1p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/1p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/1p/train_${device_id}.log
fi

