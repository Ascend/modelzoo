#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash scipts/run_standalone_td.sh"
echo "for example: bash scipts/run_standalone_td.sh"
echo "=============================================================================================================="

# mkdir -p ms_log
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
# export GLOG_log_dir=${CUR_DIR}/ms_log
# export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_task_distill.py \
    --device_target="Ascend" \
    --device_id=0 \
    --do_train="true" \
    --do_eval="true" \
    --td_phase1_epoch_size=5 \
    --td_phase2_epoch_size=3 \
    --task_name="SST-2" \
    --do_shuffle="true" \
    --enable_data_sink="true" \
    --data_sink_steps=100 \
    --save_ckpt_step=100 \
    --max_ckpt_num=1 \
    --load_teacher_ckpt_path="/home/admin/code/tinybert/bert_base_finetune_ascend_1.1.1_sst2.ckpt" \
    --load_gd_ckpt_path="/home/admin/code/tinybert/scripts/tiny_bert_83_10000.ckpt" \
    --load_td1_ckpt_path="/home/admin/code/tinybert/scripts/eval_model_0.9039.ckpt" \
    --train_data_dir="/home/admin/dataset/SST-2/uncased/train/" \
    --eval_data_dir="/home/admin/dataset/SST-2/uncased/eval/" \
    --schema_dir="" \
    --dataset_type="tfrecord" > log.txt 2>&1 &

