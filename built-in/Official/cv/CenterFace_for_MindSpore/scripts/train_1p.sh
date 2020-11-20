#!/bin/bash
root=$PWD # your script path
pretrained_backbone=$root"/../mobilenet_v2.ckpt" # or mobilenet_v2-b0353104.ckpt
env_sh_path=$root"/env.sh"
launch_path=$root/../src/launch.py
train_script_path=$root/../train.py
dataset_path=$root/dataset/centerface
annot_path=$dataset_path/annotations/train.json
img_dir=$dataset_path/images/train/images
server_id="127.0.0.1"

python $launch_path --nproc_per_node=1 --visible_devices="0" --server_id=$server_id --env_sh=$env_sh_path \
$train_script_path --lr=5e-4 --per_batch_size=8 --is_distributed=0 --T_max=140 \
--max_epoch=140 --warmup_epochs=0 --lr_scheduler=multistep --lr_epochs=90,120 \
--weight_decay=0.0000 --loss_scale=1024 --pretrained_backbone=$pretrained_backbone \
--data_dir=$dataset_path --annot_path=$annot_path --img_dir=$img_dir
