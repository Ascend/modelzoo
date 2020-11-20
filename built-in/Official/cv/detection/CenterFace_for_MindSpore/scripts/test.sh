#!/bin/bash
ckpt="0-125_24750.ckpt" # the model saved for epoch=125
root=$PWD
model_path=$root"/model/"
env_sh_path=$root"/env.sh"
launch_path=$root/../src/launch.py
test_script_path=$root/../test.py
dataset_path=$root/dataset
data_dir=$dataset_path/centerface/images/val/images/
save_path=$root/output/centerface/999
server_id="127.0.0.1"
ground_truth_mat=$dataset_path/centerface/ground_truth/val.mat

device_phy_id=0

python $launch_path --nproc_per_node=1 \
--visible_devices=$device_phy_id --server_id=$server_id --env_sh=$env_sh_path \
$test_script_path --is_distributed=0 --data_dir=$data_dir --test_model=$model_path \
--ground_truth_mat=$ground_truth_mat --save_dir=$save_path --rank=$device_phy_id --ckpt_name=$ckpt
