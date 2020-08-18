#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

device_id=$1
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
mkdir -p ${currentDir}/${device_id}
rm -rf ${currentDir}/${device_id}/*
cd ${currentDir}/${device_id}

#start exec
python3.7 /code/bert-nv-npu/run_pretraining.py --bert_config_file=/code/bert-nv-npu/configs/nezha_config_large_cn.json --max_seq_length=128 --max_predictions_per_seq=20 --train_batch_size=64 --learning_rate=1e-4 --num_warmup_steps=10000 --num_train_steps=1000000 --optimizer_type=lamb --manual_fp16=True --use_fp16_cls=True --input_files_dir=/data/dataset/cn-news-128-100f --eval_files_dir=/data/dataset/nv-en-wiki-f16 --init_checkpoint=/data/ckpt/model.ckpt-40300 --npu_bert_debug=False --npu_bert_use_tdt=True --do_train=True --num_accumulation_steps=1 --npu_bert_job_start_file= --iterations_per_loop=100 --save_checkpoints_steps=10000 --npu_bert_clip_by_global_norm=False --distributed=True --npu_bert_loss_scale=0 --output_dir=/d_solution/ckpt${DEVICE_ID} > ${currentDir}/train_${device_id}.log 2>&1
if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
fi
