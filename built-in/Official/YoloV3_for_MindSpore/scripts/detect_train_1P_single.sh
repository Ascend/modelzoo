#!/bin/bash
dos2unix /path/to/modelzoo_yolo/scripts/env.sh

python /path/to/modelzoo_yolo/launch.py --nproc_per_node=1 \
--visible_devices="0" --server_id="xx.xxx.xxx.xxx" --env_sh="/path/to/modelzoo_yolo/scripts/env.sh" \
/path/to/modelzoo_yolo/train.py --lr=0.0125 --per_batch_size=32 \
--is_distributed=0 --T_max=320 --max_epoch=320 --warmup_epochs=4 --lr_scheduler=cosine_annealing --data_dir=/path/to/dataset \
--pretrained_backbone=/path/to/modelzoo_yolo/models/0-148_92000.ckpt --training_shape=416

