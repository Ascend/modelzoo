DATA_PATH="../wmt-ende"
TEST_SOURCES="${DATA_PATH}/tfrecord/newstest2014.l128.tfrecord"
MODEL_DIR="file://PATH_TO_BE_CONFIGURED"
export JOB_ID=10086
export DEVICE_ID=2
export RANK_ID=0
export RANK_SIZE=1

python3.7 -m noahnmt.bin.infer \
  --tasks="
    - class: decode_text
      params:
        output: ./output-0603
        output_nbest: False
        unk_replace: False
        postproc_fn: null" \
  --model_params="
    vocab_source: ${DATA_PATH}/vocab.share
    vocab_target: ${DATA_PATH}/vocab.share
    decoder.params:
      position.max_length: 80
    inference.use_sampling: False
    inference.beam_search.beam_width: 4
    inference.beam_search.length_penalty_weight: 1.0" \
  --input_pipeline="
    class: parallel_tfrecord_input_pipeline
    params:
      fix_batch: true
      max_length: 128
      files: ${TEST_SOURCES}" \
  --enable_graph_rewriter False \
  --batch_size 1 \
  --model_dir ${MODEL_DIR} --use_fp16=True
