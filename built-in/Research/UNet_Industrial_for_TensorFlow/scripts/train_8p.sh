#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
#echo ${currentDir}
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")
RES=$4

device_id=$3
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${RES}/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi
export RANK_ID=${device_id}


DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path

RES_DIR=${RES}/${device_id}
mkdir -p ${RES_DIR}
rm -rf ${RES_DIR}/*


#start exec
python3 ${currentDir}/../main.py \
    --unet_variant='tinyUNet' \
    --activation_fn='relu' \
    --exec_mode='train_and_evaluate' \
    --iter_unit='batch' \
    --num_iter=2500 \
    --batch_size=2 \
    --warmup_step=10 \
    --results_dir="${RES_DIR}/" \
    --data_dir="${1}" \
    --dataset_name='DAGM2007' \
    --dataset_classID="${2}" \
    --data_format='NCHW' \
    --use_auto_loss_scaling \
    --nouse_tf_amp \
    --nouse_xla \
    --learning_rate=1e-4 \
    --learning_rate_decay_factor=0.8 \
    --learning_rate_decay_steps=500 \
    --rmsprop_decay=0.9 \
    --rmsprop_momentum=0.8 \
    --loss_fn_name='adaptive_loss' \
    --weight_decay=1e-5 \
    --weight_init_method='he_uniform' \
    --augment_data \
    --display_every=250 \
    --debug_verbosity=0 > ${RES}/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${RES}/train_${device_id}.log
else
    echo "turing train fail" >> ${RES}/train_${device_id}.log
fi


