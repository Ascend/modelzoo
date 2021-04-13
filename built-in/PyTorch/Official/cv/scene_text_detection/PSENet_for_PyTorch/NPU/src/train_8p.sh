#!/usr/bin/env bash

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

taskset -c 0-95 python3.7 -W ignore train_ic15_8p.py \
  --lr 0.004\
	--dist-backend 'hccl' \
	--rank 0  \
	--workers 32 \
	--multiprocessing-distributed \
	--world-size 1 \
	--batch_size 32 \
	--device 'npu' \
	--opt-level 'O2' \
	--loss-scale 64 \
	--combine_grad \
	--combine_sgd \
	--addr='XX.XXX.XXX.XXX' \
	--seed 16  \
	--n_epoch 600 \
	--data-dir '/home/data/' \
	--port 8272 \
	--device-list '0,1,2,3,4,5,6,7' \
	--remark 'test'
