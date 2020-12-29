#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID))
export DEVICE_INDEX=${DEVICE_INDEX}

# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"

#mkdir exec path
mkdir -p ${currentDir}/result/8p/${device_id}
cd ${currentDir}/result/8p/${device_id}

python3.7 ${currentDir}/main_npu.py \
    --data_dir=/data/slimImagenet \
    --model_dir=./ \
    --mode=train_and_eval \
    --train_batch_size=256 \
    --train_steps=218750 \
    --iterations_per_loop=625 \
    --steps_per_eval=31250 \
    --base_learning_rate=0.2 \
    --model_name=efficientnet-b0 > ${currentDir}/result/8p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/8p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
fi

