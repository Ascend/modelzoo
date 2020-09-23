#!/bin/bash

train_path=/PATH/TO/EXPERIMENTS_DIR
export SLOG_PRINT_TO_STDOUT=0
train_code_path=/PATH/TO/MODEL_ZOO_CODE
export RANK_TABLE_FILE=${train_code_path}/tools/rank_table_8p.json
export RANK_SIZE=8
export RANK_START_ID=0
rm -rf ${train_path}
mkdir -p ${train_path}
mkdir ${train_path}/ckpt

ulimit -c unlimited

for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train_seg_multicards.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=32  \
                                               --crop_size=513  \
                                               --base_lr=0.08  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s16  \
                                               --freeze_bn=False  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --save_steps=500  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
