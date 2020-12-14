#!/bin/bash

if [ ! $1 ]; then
    $1='ppi'
fi
if [ ! $2 ]; then
    $2='meanpool'
fi

python3 graphsage/supervised_train.py --train_prefix data/$1/$1 --base_log_dir outputs/$1 --model $2 --device gpu --gpu_ids 0
