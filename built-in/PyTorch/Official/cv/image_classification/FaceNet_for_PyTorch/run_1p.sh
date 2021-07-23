#!/usr/bin/env bash
source ./env_b031.sh
source ./env_new.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 1
/usr/local/Ascend/driver/tools/msnpureport -g error -d 2
/usr/local/Ascend/driver/tools/msnpureport -g error -d 3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 4
/usr/local/Ascend/driver/tools/msnpureport -g error -d 5
/usr/local/Ascend/driver/tools/msnpureport -g error -d 6
/usr/local/Ascend/driver/tools/msnpureport -g error -d 7
/usr/local/Ascend/driver/tools/msnpureport -e disable

python3.7 fine_tune_new.py \
    --seed 12345 \
    --amp_cfg \
    --opt_level O2 \
    --loss_scale_value 1024 \
    --device_list '0' \
    --batch_size 512 \
    --epochs 8 \
    --epochs_per_save 1 \
    --lr 0.001 \
    --workers 8 \
    --data_dir '/home/VGG-Face2/data/train_cropped'
