#!/bin/bash

#if [ ! -d "result/8p" ]; then
#    mkdir -p result/8p
#else
#    rm -rf result/8p/*
#fi
mkdir -p result/4p

currentDir=$(cd "$(dirname "$0")"; pwd)

# 8p
device_id=0,1,2,3

export CUDA_VISIBLE_DEVICES=${device_id}

#mkdir -p ${currentDir}/result/8p
#rm -rf ${currentDir}/result/8p/*
#cd ${currentDir}/result/8p
mkdir -p results

export TF_ENABLE_AUTO_MIXED_PRECISION=1

horovodrun -np 4 -H localhost:4 python ${currentDir}/train_image_classifier_yz.py \
       --augment_images=True \
       --dataset_dir=data/ilsvrc2012_tfrecord \
       --dataset_split_name=train \
       --enable_hvd=True \
       --enable_apex=False \
       --gpu_ids='0,1,2,3' \
       --label_smoothing=0.1 \
       --learning_rate_decay_factor=0.99 \
       --msg='me_param' \
       --model_name="mobilenet_v3_large" \
       --moving_average_decay=0.9999 \
       --num_epochs_per_decay=1.0 \
       --preprocessing_name="inception_v2" \
       --weight_decay='0.00004' \
       --batch_size=320 \
       --learning_rate_decay_type='cosine_annealing' \
       --learning_rate=0.3 \
       --optimizer='momentum' \
       --momentum='0.9' \
       --max_epoch=1200 \
       --measure_accu_during_train='True' \
       --run_mode='estimator' \
       --data_loader_mode='united' \
       --warmup_epochs=4 2>&1 | tee ${currentDir}/result/4p/train.log 


wait
