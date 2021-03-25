set -x
export POD_NAME=another0
execpath=${PWD}
rm -rf *.pbtxt
ulimit -c 0

python3.7 -m run_squad_v2 \
  --output_dir=./output_large_v2 \
  --input_dir=./squad_v2 \
  --model_dir=./albert_large_v2 \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train \
  --do_predict \
  --train_batch_size=16 \
  --predict_batch_size=8 \
  --learning_rate=1.5e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=500 \
  --n_best_size=20 \
  --max_answer_length=30
  #--model_dir=./model_path \
