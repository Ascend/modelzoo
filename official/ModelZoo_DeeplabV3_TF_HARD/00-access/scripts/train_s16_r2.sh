#!/bin/sh


CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

cd ..
CURRENT_DIR=$( pwd)


PWD=${CURRENT_DIR}

DEVICE_ID=$1
if  [ x"${DEVICE_ID}" = x ] ;
then
    mkdir -p ${CURRENT_DIR}/results/s8
    echo "turing train fail" >> ${CURRENT_DIR}/results/s16/train_${DEVICE_ID}.log
    exit
else
    export DEVICE_ID=${DEVICE_ID}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${DEVICE_ID}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

RESULTS=results/s16/r2
#mkdir exec path
mkdir -p ${CURRENT_DIR}/${RESULTS}/${DEVICE_ID}
rm -rf ${CURRENT_DIR}/${RESULTS}/${DEVICE_ID}/*
cd ${CURRENT_DIR}/${RESULTS}/${DEVICE_ID}


CKPT_NAME=${CURRENT_DIR}/results/s16/r1/0/resnet_101/model.ckpt-15000
PASCAL_DATASET=${CURRENT_DIR}/datasets/pascal_voc_seg/tfrecord

NUM_ITERATIONS=7000
#start exec
python3.7 ${CURRENT_DIR}/train_npu.py --model_variant='resnet_v1_101' \
                                      --train_split='train' \
                                      --atrous_rates=6 \
                                      --atrous_rates=12 \
                                      --atrous_rates=18 \
                                      --output_stride=16 \
                                      --train_crop_size="513,513" \
                                      --train_batch_size=8 \
                                      --training_number_of_steps="${NUM_ITERATIONS}" \
                                      --fine_tune_batch_norm=true \
                                      --tf_initial_checkpoint="${CKPT_NAME}" \
                                      --log_steps=100 \
                                      --weight_decay=0.00002 \
                                      --last_layer_gradient_multiplier=1 \
                                      --bias_multiplier=2.0 \
                                      --rank=8 \
                                      --multi_grid=1 \
                                      --multi_grid=2 \
                                      --multi_grid=4 \
                                      --display_every=500 \
                                      --iterations_per_loop=500 \
                                      --aspp_with_separable_conv=False \
                                      --learning_policy="cosine" \
                                      --base_learning_rate=0.0004 \
                                      --decay_steps="${NUM_ITERATIONS}" \
                                      --dataset_dir="${PASCAL_DATASET}"> ./train_${DEVICE_ID}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ./train_${DEVICE_ID}.log
else
    echo "turing train fail" >> ./train_${DEVICE_ID}.log
fi

