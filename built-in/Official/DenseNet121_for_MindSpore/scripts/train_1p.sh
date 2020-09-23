#!/bin/bash

python /path/to/modelzoo_densenet121/launch.py \
--nproc_per_node=1 --visible_devices=0 \
--env_sh=/path/to/modelzoo_densenet/scripts/env.sh \
--server_id=xx.xxx.xxx.xxx  /path/to/modelzoo_densenet121/train.py  \
--data_dir /path/to/dataset/train/ \
--per_batch_size 32 --max_epoch 120 --lr 0.0125 \
--lr_scheduler cosine_annealing --is_distributed 0


