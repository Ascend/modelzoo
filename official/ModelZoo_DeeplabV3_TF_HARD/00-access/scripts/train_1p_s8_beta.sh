#!/bin/sh
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

cd ..

EXEC_DIR=$( pwd)

echo ${EXEC_DIR}


PWD=${CURRENT_DIR}

RESULTS=results/1p/s8
mkdir -p  ${EXEC_DIR}/${RESULTS}

DEVICE_ID=$1
if  [ x"${DEVICE_ID}" = x ] ;
then
    export DEVICE_ID=0
    echo "Device id not specified from terminal , use device id =0 by default" >> ${CURRENT_DIR}/${RESULTS}/train_${DEVICE_ID}.log
    
else
    export DEVICE_ID=${DEVICE_ID}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${DEVICE_ID}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

DO_CHECKPOINT=True

#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}
rm -rf ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}/*
cd ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}

CKPT_NAME=${EXEC_DIR}/pretrained/model.ckpt
PASCAL_DATASET=${EXEC_DIR}/datasets/pascal_voc_seg/tfrecord

NUM_ITERATIONS=1000
#start exec
python3.7 ${EXEC_DIR}/train_npu.py --model_variant='resnet_v1_101_beta' \
                                    --train_split='trainaug' \
                                    --atrous_rates=6 \
                                    --atrous_rates=12 \
                                    --atrous_rates=18 \
                                    --output_stride=8 \
                                    --train_crop_size="513,513" \
                                    --train_batch_size=16 \
                                    --training_number_of_steps="${NUM_ITERATIONS}" \
                                    --fine_tune_batch_norm=true \
                                    --tf_initial_checkpoint="${CKPT_NAME}" \
                                    --log_steps=100 \
                                    --weight_decay=0.0001 \
                                    --last_layer_gradient_multiplier=1 \
                                    --bias_multiplier=1.0 \
                                    --rank=1 \
                                    --do_checkpoint=${DO_CHECKPOINT} \
                                    --multi_grid=1 \
                                    --multi_grid=2 \
                                    --multi_grid=4 \
                                    --display_every=100 \
                                    --iterations_per_loop=100 \
                                    --aspp_with_separable_conv=False \
                                    --learning_policy="cosine" \
                                    --base_learning_rate=0.014 \
                                    --decay_steps="${NUM_ITERATIONS}" \
                                    --dataset_dir="${PASCAL_DATASET}"> ./train_${DEVICE_ID}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ./train_${DEVICE_ID}.log
else
    echo "turing train fail" >> ./train_${DEVICE_ID}.log
fi

