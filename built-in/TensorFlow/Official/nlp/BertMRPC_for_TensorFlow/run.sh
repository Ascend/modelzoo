# set env


export JOB_ID=10086
export DEVICE_ID=0
export DEVICE_INDEX=0
export RANK_ID=0
export RANK_SIZE=1

export ASCEND_GLOBAL_LOG_LEVEL=3

export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3

BERT_BASE_DIR=model
GLUE_DIR=dataset

#rm -rf output

python3 run_classifier.py \
  --task_name=MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --do_eval=True \
  --data_dir=$GLUE_DIR/MRPC \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --max_seq_length=128 \
  --output_dir=./output
