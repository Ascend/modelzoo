MODEL_DIR="./model_dir_base"


python3.7 -m noahnmt.bin.frozen_graph \
  --model_params="
    vocab_source: /home/wmt-ende/vocab.share
    vocab_target: /home/wmt-ende/vocab.share
    inference.use_sampling: False
    decoder.params:
      position.max_length: 80
    inference.beam_search.beam_width: 1
    inference.beam_search.coverage_penalty_weight: 0
    inference.beam_search.length_penalty_weight: 1.0" \
  --model_dir ${MODEL_DIR} \
  --disable_vocab_table=True \
  --optimize=True \
  --output_filename "transformer.pb" > ./frozen_graph.log 2>&1 &
