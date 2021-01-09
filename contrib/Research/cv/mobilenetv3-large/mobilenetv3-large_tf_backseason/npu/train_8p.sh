#!/bin/sh

device_id=$1
currentDir=$2
resultDir=$3
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${resultDir}/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${resultDir}/${device_id}
cd ${resultDir}/${device_id}

#python3.7 ${currentDir}/train.py \
#    --dataset_dir=/opt/npu/slimImagenet \
#    --max_epoch=300 \
#    --model_name="mobilenet_v2" \
#    --moving_average_decay=0.9999 \
#    --label_smoothing=0.1 \
#    --preprocessing_name="inception_v2" \
#    --weight_decay='0.00004' \
#    --batch_size=256 \
#    --learning_rate_decay_type='cosine_annealing' \
#    --learning_rate=0.8 \
#    --optimizer='momentum' \
#    --momentum='0.9' \
#    --warmup_epochs=5 > ${currentDir}/result/8p/train_${device_id}.log 2>&1

python3.7 ${currentDir}/train.py \
    --dataset_name=flowers \
    --dataset_dir=/cache/tf_flowers \
    --dataset_split_name=train \
    --model_name="mobilenet_v3_large" \
    --max_train_steps=2000 \
    --batch_size=256 \
    --label_smoothing=0.1 \
    --weight_decay=1e-5 \
    --moving_average_decay=0.9999 \
    --learning_rate=0.08 \
    --learning_rate_decay_factor=0.99 \
    --optimizer='rmsprop' \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --rmsprop_epsilon=0.002 \
    --init_stddev=0.008 \
    --dropout_keep_prob=0.8 \
    --bn_moving_average_decay=0.997 \
    --bn_epsilon=0.001 \
    --warmup_epochs=5 2>&1 | tee ${resultDir}/train_${device_id}.log

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${resultDir}/train_${device_id}.log
else
    echo "turing train fail" >> ${resultDir}/train_${device_id}.log
fi

