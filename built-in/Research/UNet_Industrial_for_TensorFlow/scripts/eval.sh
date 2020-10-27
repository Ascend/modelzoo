#!/bin/bash
#set env

export JOB_ID=10087
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=../config/new_rank_table_1p.json

dir=`pwd`
currentDir=$(cd "$(dirname "$0")"; pwd)

export TF_CPP_MIN_LOG_LEVEL=3

rm -rf /var/log/npu/slog
rm -f *.txt
rm -f *.pbtxt
rm -fr dump*
rm -f  *.log


BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


rm -rf /var/log/npu/slog
rm -f *.txt
rm -f *.pbtxt
rm -fr dump*
rm -f  *.log

python3 ${BASEDIR}/../main.py \
    --unet_variant='tinyUNet' \
    --activation_fn='relu' \
    --exec_mode='evaluate' \
    --iter_unit='epoch' \
    --num_iter=1 \
    --batch_size=16 \
    --warmup_step=10 \
    --results_dir="${1}" \
    --data_dir="${2}" \
    --dataset_name='DAGM2007' \
    --dataset_classID="${3}" \
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
    --display_every=50 \
    --debug_verbosity=0


