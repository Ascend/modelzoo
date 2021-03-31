#!/bin/bash

set -e
set -o pipefail

#export DUMP_GE_GRAPH=1
#export NPU_DEBUG=true
#export NPU_DUMP_GRAPH=true
#export NPU_ENABLE_PERF=true
loop_size=10000
export NPU_LOOP_SIZE=$loop_size
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error

cd ./tensorflow
rm -rf ckpt_new
find -name "ge_*" -print0 | xargs -i -0 rm -f {}
find -name "*.pbtxt" -print0 | xargs -i -0 rm -f {}
#python3 resnet_ctl_imagenet_main.py --data_dir=/autotest/x00505833/datasets/imagenet_dataset/tf_records/train --train_steps=${loop_size} --model_dir=./ckpt --distribution_strategy=off --use_tf_while_loop=true --use_tf_function=true --enable_checkpoint_and_export --steps_per_loop=${loop_size} --skip_eval

#python3 resnet_ctl_imagenet_main.py --data_dir=/home/imagenet_TF --train_epochs=1 --train_steps=${loop_size} --num_accumulation_steps=0 --model_dir=./ckpt_new --distribution_strategy=off --use_tf_while_loop=true --use_tf_function=true --enable_checkpoint_and_export --steps_per_loop=${loop_size} --skip_eval  

python3 resnet_ctl_imagenet_main.py --data_dir=/home/imagenet_TF --num_accumulation_steps=0 --train_epochs=1 --train_steps=${loop_size} --model_dir=./ckpt_new --distribution_strategy=off --use_tf_while_loop=true --use_tf_function=true --enable_checkpoint_and_export --steps_per_loop=${loop_size} --skip_eval --drop_eval_remainder=True 2>&1 | tee train_1p.log
