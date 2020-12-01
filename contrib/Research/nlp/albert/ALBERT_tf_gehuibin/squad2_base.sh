set -x
export POD_NAME=another0

export install_path=/usr/local/Ascend/nnae/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/add-ons/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:${install_path}/fwkacllib/python/site-packages/hccl:$PYTHONPATH
export ASCEND_OPP_PATH=/usr/local/Ascend/nnae/latest/opp

export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=200

export RANK_INDEX=0

export JOB_ID=10086
export PRINT_MODEL=1
export RANK_ID=another0
export RANK_SIZE=1
export DUMP_GE_GRAPH=2


execpath=${PWD}

device_phy_id=3
export DUMP_GE_GRAPH=1
export DEVICE_ID=$device_phy_id
export DEVICE_INDEX=$device_phy_id
export SLOG_PRINT_STDOUT=1

rm -rf *.pbtxt
ulimit -c 0

python3.7 -m run_squad_v2 \
  --output_dir=./output_base_v2 \
  --input_dir=./squad_v2 \
  --model_dir=./albert_base_v2 \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train \
  --do_predict \
  --train_batch_size=32 \
  --predict_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=500 \
  --n_best_size=20 \
  --max_answer_length=30
  #--model_dir=./model_path \
