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

export SLOG_PRINT_TO_STDOUT=1

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${resultDir}/${device_id}/results
cd ${resultDir}/${device_id}
cp ${currentDir}/snapshots/* results/

python3.7 ${currentDir}/train.py \
    --dataset_name=imagenet \
    --dataset_dir=/cache/ilsvrc2012_tfrecord \
    --dataset_split_name=train \
    --model_name="mobilenet_v3_large" \
    --max_epoch=600 \
    --moving_average_decay=0.9999 \
    --iterations_per_loop=50 \
    --batch_size=640 \
    --learning_rate_decay_type='cosine_annealing' \
    --label_smoothing=0.1 \
    --weight_decay='0.00004' \
    --preprocessing_name="inception_v2" \
    --learning_rate=0.15 \
    --learning_rate_decay_factor=0.99 \
    --optimizer='momentum' \
    --momentum='0.9' \
    --num_epochs_per_decay=2.0 \
    --warmup_epochs=4 2>&1 | tee ${resultDir}/train_${device_id}.log

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${resultDir}/train_${device_id}.log
else
    echo "turing train fail after run train" >> ${resultDir}/train_${device_id}.log
fi

