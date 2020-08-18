python ../src/downstream/run_classifier.py \
  --task_name=/your/task/name/xnli/chnsenti/or/lcqmc \
  --data_dir=/your/data/path \
  --vocab_file=../src/configs/nezha_large_vocab.txt \
  --bert_config_file=../src/configs/nezha_large_config.json \
  --init_checkpoint=/your/pretrained/model/path/ \
  --output_dir=/your/output/path/
