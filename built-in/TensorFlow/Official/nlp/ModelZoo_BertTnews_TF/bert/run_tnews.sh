#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/data/wlx/Bert_Finetune/code/models
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600
export JOB_ID=job9999001
export RANK_TABLE_FILE=/data/wlx/Bert_Finetune/result/cloud-localhost-20200418093529-0/config/hccl.json
export RANK_SIZE=1
export RANK_INDEX=0
export DEVICE_ID=0
export RANK_ID=cloud-localhost-20200418093529-0

python3 run_classifier.py --task_name=tnews --do_train=true --do_eval=true --data_dir=/data/wlx/Bert_Finetune//data/tnews --vocab_file=/data/wlx/Bert_Finetune/data/prev_trained_model/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/data/wlx/Bert_Finetune/data/prev_trained_model/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data/wlx/Bert_Finetune/data/prev_trained_model/chinese_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./
