python ../src/downstream/run_classifier.py \
  --task_name=/your/task/name/xnli/chnsenti/or/lcqmc \
  --data_dir=/your/data/path \
  --vocab_file=../src/configs/bert_base_vocab.txt \
  --bert_config_file=../src/configs/bert_base_config.json \
  --init_checkpoint=/your/pretrained/model/path/ \
  --output_dir=/your/output/path/
