#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

# user env
export JOB_ID=bert-base-1p
#export RANK_TABLE_FILE=../configs/1p.json
export RANK_SIZE=1
export RANK_INDEX=0
export RANK_ID=0

PWD=${currentDir}

device_id=0
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

env > ${currentDir}/env_${device_id}.log

#mkdir exec path
#mkdir -p ${currentDir}/${device_id}
#rm -rf ${currentDir}/${device_id}/*
cd ${currentDir}/
rm -rf kernel_meta
rm -rf output
#start exec
python3.7 ../src/pretrain/run_pretraining.py --bert_config_file=../configs/bert_base_config.json --max_seq_length=128 --max_predictions_per_seq=20 --train_batch_size=128 --learning_rate=1e-4 --num_warmup_steps=10000 --num_train_steps=500000 --optimizer_type=adam --manual_fp16=True --use_fp16_cls=True --input_files_dir=/home/models/dataset/wikipedia_128 --eval_files_dir=/autotest/CI_daily/Bert_NV/data/dataset/cn-wiki-128 --npu_bert_debug=False --npu_bert_use_tdt=True --do_train=True --num_accumulation_steps=1 --npu_bert_job_start_file= --iterations_per_loop=100 --save_checkpoints_steps=10000 --npu_bert_clip_by_global_norm=False --distributed=False --npu_bert_loss_scale=0 --output_dir=./output
