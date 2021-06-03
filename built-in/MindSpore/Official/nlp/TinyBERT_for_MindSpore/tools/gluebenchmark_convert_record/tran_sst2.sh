export BERT_BASE_DIR=/home/admin/dataset/uncased_L-12_H-768_A-12
export GLUE_DIR=/home/admin/dataset/SST-2

python run_classifier.py \
  --task_name=SST2 \
  --data_dir=$GLUE_DIR/SST-2 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --output_dir=./SST-2/uncased/ &
