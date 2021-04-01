#!/bin/bash
#export RANK_ID=$1
#export RANK_SIZE=$2
#export DEVICE_ID=$RANK_ID
#export DEVICE_INDEX=$RANK_ID
#export JOB_ID=990
#export FUSION_TENSOR_SIZE=1000000000
# for producible results
#export TF_DETERMINISTIC_OPS=1
#export TF_CUDNN_DETERMINISM=1
upDir=$(dirname "$PWD")

python3 $upDir/main.py --data_dir $upDir/data --model_dir $upDir/results --log_every 100 --max_steps 1000 --batch_size $1 --exec_mode train --augment --benchmark --npu_loss_scale 1 2>&1 | tee $upDir/results/train_$2.log

sleep 2
echo "**************** train finished ***************"
cp -r /root/ascend/log ./slog

