#!/bin/bash

python /opt/npu/mxk/densenet121/launch.py \
--nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/opt/npu/mxk/densenet121/scripts/env.sh \
--server_id=10.155.111.159 \
/opt/npu/mxk/densenet121/test.py  \
--data_dir /opt/npu/datasets/imagenet/val/ \
--per_batch_size 32 --pretrained /opt/npu/mxk/densenet121/run_test/train_res/device0/outputs/2020-04-09_time_20_40_48

