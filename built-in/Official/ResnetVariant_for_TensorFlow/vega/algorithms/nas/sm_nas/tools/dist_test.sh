#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
