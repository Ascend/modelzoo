#!/bin/bash
dir=`pwd`

# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
export ASCEND_DEVICE_ID=0
#export TUNE_OPS_NAME=
#export REPEAT_TUNE=True
#export ENABLE_TUNE_BANK=True
mkdir -p $TUNE_BANK_PATH
chown -R HwHiAiUser:HwHiAiUser $TUNE_BANK_PATH

# DataDump
export FLAG_ENABLE_DUMP=False
export DUMP_PATH=/var/log/npu/dump
export DUMP_STEP="0|2"
export DUMP_MODE="all"
mkdir -p $DUMP_PATH
chown -R HwHiAiUser:HwHiAiUser $DUMP_PATH

export JOB_ID=10086
export DEVICE_ID=2
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=${dir}/new_rank_table_1p.json
cd ../
sh train-ende.sh