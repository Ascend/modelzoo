#!/bin/bash

EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd $EXEC_DIR
echo $EXEC_DIR

rm -rf /var/log/npu/slog/host-0/*

source ${EXEC_DIR}/env.sh

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export DEVICE_ID=0
export RANK_ID=npu1p
export RANK_TABLE_FILE=${EXEC_DIR}/1p.json

export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3

cd ..

python3 ${EXEC_DIR}/../BigGAN/main.py --phase test --dataset /home/models/ModelZoo_BigGAN_TF_new/dataset/train --checkpoint_dir /home/models/ModelZoo_BigGAN_TF_new/results/8p/0/checkpoint --epoch 10000 --iteration 100 --batch_size 64 --g_lr 0.0002 --d_lr 0.0002 --img_size 128
