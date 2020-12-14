#!/usr/bin/env bash

# source env.sh

su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\""
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 0"
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 4"

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1

python3.7 -W ignore train_ic15.py \
  --lr 0.001\
	--dist-backend 'hccl' \
	--rank 0  \
	--workers 32 \
	--multiprocessing-distributed \
	--world-size 1 \
	--batch_size 16 \
	--device 'npu' \
	--opt-level 'O2' \
	--loss-scale 64 \
	--addr='XX.XXX.XXX.XXX' \
	--seed 16  \
	--n_epoch 600 \
	--data-dir '/home/data/' \
	--port 8272 \
	--schedule 200 400 \
	--device-list '1' \
	--remark 'test'
