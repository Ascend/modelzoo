#!/bin/bash
python /path/to/modelzoo_vgg16/launch.py \
--nproc_per_node=8 \
--visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/path/to/modelzoo_vgg16/scripts/env.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/modelzoo_vgg16/train.py \
--per_batch_size=32 \
--data_dir=/path/to/dataset/train/ \
--is_distributed=1 \
--backbone=vgg16  \
--lr=0.01 \
--lr_scheduler=cosine_annealing \
--max_epoch=150 \
--weight_decay=0.0001 \
--label_smooth=1
