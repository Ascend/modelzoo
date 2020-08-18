python ../src/downstream/run_ner.py \
  --data_dir=/your/data/path \
  --vocab_file=../src/configs/nezha_large_vocab.txt \
  --bert_config_file=../src/configs/nezha_large_config.json \
  --init_checkpoint=/your/pretrained/model/path/ \
  --output_dir=/your/output/path/
  
