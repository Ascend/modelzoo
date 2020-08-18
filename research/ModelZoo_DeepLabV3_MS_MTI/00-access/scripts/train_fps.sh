#!/bin/bash

export DEVICE_ID=5
export SLOG_PRINT_TO_STDOUT=0
train_path=/PATH/TO/EXPERIMENTS_DIR
train_code_path=/PATH/TO/MODEL_ZOO_CODE
rm -rf ${train_path}
mkdir -p ${train_path}
mkdir ${train_path}/device$DEVICE_ID
mkdir ${train_path}/ckpt
cd ${train_path}/device$DEVICE_ID

python ${train_code_path}/train_seg.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=50  \
                    --batch_size=8  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --freeze_bn=False  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200