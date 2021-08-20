#!/usr/bin/env bash
source env_npu.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export HCCL_WHITELIST_DISABLE=1

/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 1
/usr/local/Ascend/driver/tools/msnpureport -g error -d 2
/usr/local/Ascend/driver/tools/msnpureport -g error -d 3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 4
/usr/local/Ascend/driver/tools/msnpureport -g error -d 5
/usr/local/Ascend/driver/tools/msnpureport -g error -d 6
/usr/local/Ascend/driver/tools/msnpureport -g error -d 7
/usr/local/Ascend/driver/tools/msnpureport -e disable

python3.7 fine_tune_new_8p.py \
    --seed 12345 \
    --amp_cfg \
    --opt_level O2 \
    --loss_scale_value 1024 \
    --device_list '0,1,2,3,4,5,6,7' \
    --batch_size 4096 \
    --epochs 8 \
    --epochs_per_save 1 \
    --lr 0.005 \
    --workers 64 \
    --data_dir '/home/VGG-Face2/data/train_cropped' \
    --addr=$(hostname -I |awk '{print $1}') \
    --rank 0 \
    --dist_url 'tcp://127.0.0.1:50000' \
    --dist_backend 'hccl' \
    --multiprocessing_distributed \
    --world_size 1
