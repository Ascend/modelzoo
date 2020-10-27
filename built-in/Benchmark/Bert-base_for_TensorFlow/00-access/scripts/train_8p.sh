#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi


DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}
export RANK_ID=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/8p/${device_id}
rm -rf ${currentDir}/result/8p/${device_id}/*
cd ${currentDir}/result/8p/${device_id}

#start exec
python3.7 ${dname}/src/pretrain/run_pretraining.py --bert_config_file=${dname}/configs/bert_base_config.json --max_seq_length=128 --max_predictions_per_seq=20 --train_batch_size=128 --learning_rate=1e-4 --num_warmup_steps=10000 --num_train_steps=500000 --optimizer_type=adam --manual_fp16=True --use_fp16_cls=True --input_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 --eval_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 --npu_bert_debug=False --npu_bert_use_tdt=True --do_train=True --num_accumulation_steps=1 --npu_bert_job_start_file= --iterations_per_loop=100 --save_checkpoints_steps=10000 --npu_bert_clip_by_global_norm=False --distributed=True --npu_bert_loss_scale=0 --output_dir=./output > ${currentDir}/result/8p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/8p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
fi

