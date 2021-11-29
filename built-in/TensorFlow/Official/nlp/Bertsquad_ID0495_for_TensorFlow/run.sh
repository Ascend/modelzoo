export install_path=/usr/local/Ascend
# driver包依赖
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
#fwkacllib 包依赖
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:{install_path}/fwkacllib/bin:$PATH
#tfplugin 包依赖
export PYTHONPATH=/usr/local/Ascend/tfplugin/python/site-packages:$PYTHONPATH
# opp包依赖
export ASCEND_OPP_PATH=${install_path}/opp


export JOB_ID=10086
#export DEVICE_ID=0
#export DEVICE_INDEX=0
#export RANK_ID=0
export RANK_SIZE=1

#export ASCEND_SLOG_PRINT_TO_STDOUT=3
#export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3

BERT_BASE_DIR=model
SQUAD_DIR=dataset

#rm -rf output

python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=None
