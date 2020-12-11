#!/usr/bin/env bash

# source env.sh

su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\""
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 0"
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 4"

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1

python3 -W ignore train_8p_anycard.py \
  --lr 0.008\
	--dist-backend 'hccl' \
	--rank 0  \
	--workers 32 \
	--multiprocessing-distributed \
	--world-size 1 \
	--batch_size 32 \
	--device 'npu' \
	--opt-level 'O2' \
	--loss-scale 64 \
	--addr='10.246.246.76' \
	--seed 16  \
	--n_epoch 600 \
	--data-dir '/home/data/' \
	--port 8271 \
	--schedule 200 400 \
	--device-list '0,1,2,3,4,5,6,7' \
	--remark 'npu8pbatch32lr8'
# --resume "/home/z00524916/deploy/8p/best/npu8pscheule_0.3868_0.9474_0.8690_0.9098_588.pth"
# --resume 'best/0.4274_0.9001_0.8578_146.pth'\
