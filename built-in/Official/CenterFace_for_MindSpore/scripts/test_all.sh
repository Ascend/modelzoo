#!/bin/bash
root=$PWD
model_path=$root"/model/"
env_sh_path=$root"/env.sh"
launch_path=$root/../src/launch.py
test_script_path=$root/../test.py
dataset_path=$root/dataset
data_dir=$dataset_path/centerface/images/val/images/
save_path=$root/output/centerface/
server_id="127.0.0.1"
ground_truth_mat=$dataset_path/centerface/ground_truth/val.mat

# blow are used for calculate model name
# model/ckpt name is "0-" + str(ckpt_num) + "_" + str(198*ckpt_num) + ".ckpt";
# ckpt_num is epoch number, can be calculated by device_num
# detail can be found in "test.py"
device_num=8
steps_per_epoch=198 #198 for 8P; 1583 for 1p
start=11 # start epoch number = start * device_num + min(device_phy_id) + 1
end=18 # end epoch number = end * device_num + max(device_phy_id) + 1

for device_phy_id in {0..7}
do
    python $launch_path --nproc_per_node=1 \
    --visible_devices=$device_phy_id --server_id=$server_id --env_sh=$env_sh_path \
    $test_script_path --is_distributed=0 --data_dir=$data_dir --test_model=$model_path \
    --ground_truth_mat=$ground_truth_mat --save_dir=$save_path --rank=$device_phy_id \
    --device_num=$device_num --steps_per_epoch=$steps_per_epoch --start=$start --end=$end
done
