#!/bin/sh

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p
export SLOG_PRINT_TO_STDOUT=0


CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

PWD=${CURRENT_DIR}

cd ..
CURRENT_DIR=$( pwd)

DEVICE_ID=1
export DEVICE_ID=${DEVICE_ID}


echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID


EXPORT_PATH=pb_model/s8
#mkdir exec path
mkdir -p ${CURRENT_DIR}/${EXPORT_PATH}
cd ${CURRENT_DIR}/${EXPORT_PATH}

CHECKPOINT_PATH=${CURRENT_DIR}/results/s8/r2/0/resnet_101/model.ckpt-10000

#start exec
python3.7 ${CURRENT_DIR}/export_model.py --checkpoint_path=${CHECKPOINT_PATH} \
                                    --model_variant="resnet_v1_101_beta" \
				    --export_path=${EXPORT_PATH} \
                                    --atrous_rates=6 \
                                    --atrous_rates=12 \
                                    --atrous_rates=18 \
                                    --output_stride=8 \
                                    --multi_grid=1 \
                                    --multi_grid=2 \
                                    --multi_grid=4 \
                                    --aspp_with_separable_conv=False 

if [ $? -eq 0 ] ;
then
    echo `date`>> ${CURRENT_DIR}/${EXPORT_PATH}/convert.log
    echo "turing train success" >> ${CURRENT_DIR}/${EXPORT_PATH}/convert.log
else
    echo `date`>> ${CURRENT_DIR}/${EXPORT_PATH}/convert.log
    echo "turing train failure" >> ${CURRENT_DIR}/${EXPORT_PATH}/convert.log
fi
