#!/bin/sh





CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p

export SLOG_PRINT_TO_STDOUT=0


CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

cd ..
CURRENT_DIR=$( pwd)

DEVICE_ID=0
export DEVICE_ID=${DEVICE_ID}


echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

#mkdir exec path
mkdir -p ${CURRENT_DIR}/eval/${DEVICE_ID}
cd ${CURRENT_DIR}/eval/${DEVICE_ID}

PASCAL_DATASET=${CURRENT_DIR}/datasets/pascal_voc_seg/tfrecord
CHECKPOINT_DIR=${CURRENT_DIR}/results/s16/r2/0/resnet_101/

#start exec
python3.7 ${CURRENT_DIR}/train_npu.py --mode=evaluate \
                                    --eval_split="val" \
                                    --model_variant="resnet_v1_101_beta" \
                                    --iterations_per_loop=1 \
                                    --atrous_rates=6 \
                                    --atrous_rates=12 \
                                    --atrous_rates=18 \
                                    --output_stride=16 \
                                    --multi_grid=1 \
                                    --multi_grid=2 \
                                    --multi_grid=4 \
                                    --eval_crop_size="513,513" \
                                    --aspp_with_separable_conv=False \
                                    --checkpoint_dir="${CHECKPOINT_DIR}" \
                                    --dataset_dir="${PASCAL_DATASET}" \
                                    --max_number_of_evaluations=1 > ./test_${DEVICE_ID}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo `date`>> ${CURRENT_DIR}/eval/train_${DEVICE_ID}.log
    echo "turing train success" >> ${CURRENT_DIR}/eval/train_${DEVICE_ID}.log
else
    echo `date`>> ${CURRENT_DIR}/eval/train_${DEVICE_ID}.log
    echo "turing train fail" >> ${CURRENT_DIR}/eval/train_${DEVICE_ID}.log
fi

RESULT_LOG=${CURRENT_DIR}/eval/${DEVICE_ID}/resnet_101/training.log 

python3.7 ${CURRENT_DIR}/cal_miou.py --result_log=${RESULT_LOG} >>${RESULT_LOG}



