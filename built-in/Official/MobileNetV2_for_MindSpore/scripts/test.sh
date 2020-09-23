#!/bin/bash
python /path/to/launch.py --nproc_per_node=1 --visible_devices=0 --env_sh=/path/to/env.sh --mode=test\
--server_id=xx.xxx.xxx.xxx  /path/to/test.py --per_batch_size=32 --data_dir=/path/to/dataset/val/ \
--is_distributed=0 --pretrained=/path/to/ckpt