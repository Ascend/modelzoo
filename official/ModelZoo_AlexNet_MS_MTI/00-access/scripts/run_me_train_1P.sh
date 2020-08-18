#!/bin/bash
python /path/to/modelzoo_alexnet/launch.py \
--nproc_per_node=1 \
--visible_devices=0 \
--env_sh=/path/to/modelzoo_alexnet/scripts/env.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/modelzoo_alexnet/train.py \
--per_batch_size=256 \
--data_dir=/path/to/dataset/train/ \
--is_distributed=0 \
--backbone=alexnet \
--lr=0.01625