#!/bin/bash
python /path/to/modelzoo_alexnet/launch.py \
--nproc_per_node=8 \
--visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/path/to/modelzoo_alexnet/scripts/env.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/modelzoo_alexnet/train.py \
--per_batch_size=256 \
--data_dir=/path/to/dataset/train/ \
--is_distributed=1 \
--backbone=alexnet
