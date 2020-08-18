python ../src/downstream/run_reading.py \
  --vocab_file=../src/configs/bert_base_vocab.txt \
  --bert_config_file=../src/configs/bert_base_config.json \
  --init_checkpoint=/your/pretrained/model/path/ \
  --output_dir=/your/output/path/
  --train_file=../data/cmrc/new_cmrc2018_train.json \
  --predict_file=../data/cmrc/new_cmrc2018_dev.json

python ../src/downstream/reading_evaluate.py \
  --dataset_file=../data/cmrc/new_cmrc2018_dev.json \
  --prediction_folder=/your/finetuned/output/json/ \
  --output_metrics=/your/evaluate/result/file/
