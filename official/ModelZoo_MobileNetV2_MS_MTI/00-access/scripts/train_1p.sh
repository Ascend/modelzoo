#!/bin/bash
python /path/to/launch.py --nproc_per_node=1 --visible_devices=0 --env_sh=/path/to/env.sh \
--server_id=xx.xxx.xxx.xxx  /path/to/train.py --per_batch_size=256 --data_dir=/path/to/dataset/train/ \
--is_distributed=0 --lr_scheduler=cosine_annealing --weight_decay=0.00004 --lr=0.1 --T_max=200 --max_epoch=200 \
--warmup_epochs=5 --label_smooth=1