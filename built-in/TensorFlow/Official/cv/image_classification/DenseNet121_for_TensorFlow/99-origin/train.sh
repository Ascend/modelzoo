#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

#mkdir -p ${currentDir}/log

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/log/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${device_id}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

#echo "${currentDir}"

#env > ${currentDir}/env_${device_id}.log

#mkdir exec path
mkdir -p ${currentDir}/log/${device_id}
rm -rf ${currentDir}/log/${device_id}/*
cd ${currentDir}/log/${device_id}

#mkdir -p ${currentDir}/log
#cd ${currentDir}/log

#start exec
/usr/bin/python3.7 ${currentDir}/train.py --config_file densenet_config_8p_npu > ./train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/log/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/log/train_${device_id}.log
fi

