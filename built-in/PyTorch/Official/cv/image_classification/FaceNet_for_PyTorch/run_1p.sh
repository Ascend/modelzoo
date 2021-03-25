#!/usr/bin/env bash
source ./env_b031.sh
source ./env_new.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

/usr/local/Ascend/driver/tools/msnpureport -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

python3.7 fine_tune_new.py \
    --seed 12345 \
	--amp_cfg \
	--opt_level O2 \
	--loss_scale_value 1024 \
	--device_list '2' \
	--batch_size 512 \
	--epochs 1 \
	--workers 8 \
	--data_dir '/home/VGG-Face2/data/train_cropped'
