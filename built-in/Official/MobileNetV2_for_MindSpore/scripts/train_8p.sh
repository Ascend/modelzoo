#!/bin/bash
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --env_sh=/path/to/env.sh \
--server_id=xx.xxx.xxx.xxx  /path/to/train.py --per_batch_size=256 --data_dir=/path/to/dataset/train/ \
--is_distributed=1 --lr_scheduler=cosine_annealing --weight_decay=0.00004 --lr=0.8 --T_max=200 --max_epoch=200 \
--warmup_epochs=5 --label_smooth=1 --ckpt_interval=500