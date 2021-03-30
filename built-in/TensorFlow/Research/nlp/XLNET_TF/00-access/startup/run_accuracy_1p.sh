#!/bin/bash

cur_dir=$(cd "$(dirname "$0")"; pwd)
echo "$cur_dir"


python3 $cur_dir/../run_classifier.py \
  --do_train=True \
  --do_eval=False \
  --task_name=sts-b \
  --data_dir=$cur_dir/../glue_data/STS-B \
  --output_dir=$cur_dir/../proc_data/sts-b \
  --model_dir=$cur_dir/../ckpt_npu \
  --uncased=False \
  --spiece_model_file=$cur_dir/../LARGE_dir/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=$cur_dir/../LARGE_dir/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=$cur_dir/../LARGE_dir/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True


