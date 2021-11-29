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

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/8p/${device_id}
cd ${currentDir}/result/8p/${device_id}

python3.7 ${currentDir}/train.py \
    --dataset_dir=/data/imagenet_TF \
    --max_epoch=300 \
    --model_name="mobilenet_v2" \
    --moving_average_decay=0.9999 \
    --label_smoothing=0.1 \
    --preprocessing_name="inception_v2" \
    --weight_decay='0.00004' \
    --batch_size=256 \
    --learning_rate_decay_type='cosine_annealing' \
    --learning_rate=0.8 \
    --optimizer='momentum' \
    --momentum='0.9' \
    --warmup_epochs=5 | tee -a ${currentDir}/result/8p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/8p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
fi

