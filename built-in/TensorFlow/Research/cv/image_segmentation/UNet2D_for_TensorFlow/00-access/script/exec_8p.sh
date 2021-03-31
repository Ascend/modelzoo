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

python3 $upDir/main.py --data_dir $upDir/data --model_dir $upDir/results --log_every 100 --max_steps 6400 --batch_size 8 --exec_mode train_and_evaluate --crossvalidation_idx $1 --augment --npu_loss_scale 1 2>&1 | tee $upDir/results/log_TF-AMP_8GPU_fold$1.txt

sleep 2
echo "**************** train finished ***************"
cp -r /root/ascend/log ./slog

