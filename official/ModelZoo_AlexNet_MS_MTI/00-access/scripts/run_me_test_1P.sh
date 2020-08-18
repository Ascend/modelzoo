#!/bin/bash
python /path/to/modelzoo_alexnet/launch.py \
--nproc_per_node=1 \
--visible_devices=0 \
--mode=test \
--env_sh=/path/to/modelzoo_alexnet/scripts/env.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/modelzoo_alexnet/test.py \
--per_batch_size=32 \
--data_dir=/path/to/dataset/val/ \
--is_distributed=0 \
--backbone=alexnet  \
--pretrained=/path/to/ckpt
