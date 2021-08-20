# fr=zh
# en=en
# vs="bpe.59500"

# DATA_PATH="s3://mt-data/tmtdata-ph"
# VOCAB_SOURCE=${DATA_PATH}/vocab.${vs}.$fr
# VOCAB_TARGET=${DATA_PATH}/vocab.${vs}.$en
# TRAIN_SOURCES=${DATA_PATH}/train.${vs}.$fr
# TRAIN_TARGETS=${DATA_PATH}/train.${vs}.$en
# DEV_SOURCES=${DATA_PATH}/dev.${vs}.$fr
# DEV_TARGETS=${DATA_PATH}/dev.${vs}.$en

# bs=3072
# MODEL_DIR="s3://mt-models/test/transformer"
# LOG_FILE="${MODEL_DIR}/train-log.txt"


DATA_PATH="../wmt-ende"
VOCAB_SOURCE=${DATA_PATH}/vocab.share
VOCAB_TARGET=${DATA_PATH}/vocab.share
TRAIN_FILES=${DATA_PATH}/concat128/train.l128.tfrecord-001-of-016
for idx in {002..016}; do
  TRAIN_FILES="${TRAIN_FILES},${DATA_PATH}/concat128/train.l128.tfrecord-${idx}-of-016"
done
DEV_SOURCES=${DATA_PATH}/dev2010.tok.zh
DEV_TARGETS=${DATA_PATH}/dev2010.tok.en

#bs=24
bs=40
MODEL_DIR="./model_dir"
LOG_FILE="${MODEL_DIR}/train-log.txt"

if [ -e ${MODEL_DIR} ]; then
   rm -rf ${MODEL_DIR}
   rm -rf kernel_meta/
fi

if [[ $MODEL_DIR == "s3://"* ]]; then
    cd ./noah_nmt
    export S3_REQUEST_TIMEOUT_MSEC=600000
fi

python3 -m noahnmt.bin.train \
  --config_paths="
    ./configs/transformer_medium.yml" \
  --model_params="
    max_grad_norm: 5
    optimizer.name: Adam
    disable_vocab_table: true
    learning_rate.warmup_steps: 16000
    decoder.params:
      dropout_rate: 0.2
    encoder.params:
      dropout_rate: 0.2
    mixed_precision.params:
      init_loss_scale: 1024.0
      fix_loss_scale: false
    vocab_source: $VOCAB_SOURCE
    vocab_target: $VOCAB_TARGET" \
  --metrics="
    - class: perplexity
    - class: bleu" \
  --input_pipeline_train="
    class: parallel_tfrecord_input_pipeline_concat
    params:
      fix_batch: true
      max_length: 128
      files: $TRAIN_FILES" \
  --input_pipeline_dev="
    class: parallel_tfrecord_input_pipeline_concat
    params:
      max_length: 128
      files: $TRAIN_FILES" \
  --train_steps=300000 \
  --schedule=train \
  --eval_every_n_steps=5000 \
  --eval_run_autoregressive=False \
  --eval_keep_best_n=1 \
  --early_stopping_rounds=10 \
  --early_stopping_metric="loss" \
  --early_stopping_metric_minimize=True \
  --keep_checkpoint_max=20 \
  --batch_size=$bs \
  --data_parallelism=False \
  --dp_param_shard=False \
  --enable_graph_rewriter=False \
  --model_dir=$MODEL_DIR --use_fp16=True

