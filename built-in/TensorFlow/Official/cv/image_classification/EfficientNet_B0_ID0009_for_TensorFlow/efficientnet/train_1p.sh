#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/1p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
export ASCEND_DEVICE_ID=$1
#export TUNE_OPS_NAME=
#export REPEAT_TUNE=True
#export ENABLE_TUNE_BANK=True
mkdir -p $TUNE_BANK_PATH
chown -R HwHiAiUser:HwHiAiUser $TUNE_BANK_PATH

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/1p/${device_id}
cd ${currentDir}/result/1p/${device_id}

python3.7 ${currentDir}/main_npu.py \
    --data_dir=/data/slimImagenet \
    --model_dir=./ \
    --mode=train \
    --train_batch_size=256 \
    --train_steps=100 \
    --iterations_per_loop=10 \
    --model_name=efficientnet-b0 > ${currentDir}/result/1p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/1p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/1p/train_${device_id}.log
fi

