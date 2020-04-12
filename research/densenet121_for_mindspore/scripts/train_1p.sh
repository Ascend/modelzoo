#!/bin/bash

python /opt/npu/mxk/densenet121/launch.py \
--nproc_per_node=1 --visible_devices=0 \
--env_sh=/opt/npu/mxk/densenet121/scripts/env.sh \
--server_id=10.155.111.159  /opt/npu/mxk/densenet121/train.py  \
--data_dir /opt/npu/datasets/imagenet/train/ \
--per_batch_size 32 --max_epoch 120 --lr 0.0125 \
--lr_scheduler cosine_annealing --is_distributed 0


