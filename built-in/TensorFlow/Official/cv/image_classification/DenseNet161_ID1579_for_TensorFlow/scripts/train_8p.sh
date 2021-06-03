#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/8p/${device_id}
rm -rf ${currentDir}/result/8p/${device_id}/*
cd ${currentDir}/result/8p/${device_id}

#start exec
python3.7 ${dname}/train.py --rank_size=8 \
    --mode=train_and_evaluate \
    --max_epochs=150 \
    --iterations_per_loop=1000 \
    --epochs_between_evals=149 \
    --data_dir=/opt/npu/slimImagenet \
    --lr=0.1 \
    --log_dir=./model_8p \
    --log_name=densenet121_8p.log > ${currentDir}/result/8p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/8p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
fi

