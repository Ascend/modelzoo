# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#!/bin/bash
ckpt="0-125_24750.ckpt" # the model saved for epoch=125
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
ground_truth_path=$root/dataset/centerface/ground_truth
device_phy_id=0

python $launch_path --nproc_per_node=1 \
--visible_devices=$device_phy_id --server_id=$server_id --env_sh=$env_sh_path \
$test_script_path --is_distributed=0 --data_dir=$data_dir --test_model=$model_path \
--ground_truth_mat=$ground_truth_mat --save_dir=$save_path --rank=$device_phy_id --ckpt_name=$ckpt \
--eval=1 --ground_truth_path=$ground_truth_path
