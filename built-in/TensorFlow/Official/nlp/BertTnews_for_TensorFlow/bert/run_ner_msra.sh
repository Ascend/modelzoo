export JOB_ID=10086
#export PROFILING_DIR=/var/log/npu/profiling/container/0
export DEVICE_ID=0
export PROFILING_MODE=false
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=new_rank_table_1p.json
export FUSION_TENSOR_SIZE=1000000000

#export EXPERIMENTAL_DYNAMIC_PARTITION=1
# Export variables
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1
export GE_TRAIN=0
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/Ascend/tfplugin
export PATH=/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/python3.7.5/bin/sacrebleu/:$PATH
export PYTHONPATH=$PYTHONPATH:/usr/local/python3.7.5/bin/sacrebleu/
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/runtime/lib64/tbe_plugin/bert
export WHICH_OP=GEOP
export DDK_VERSION_FLAG=1.60.T17.B830
export NEW_GE_FE_ID=1
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/Ascend/runtime/lib64/plugin/opskernel/librts_engine.so
export OP_PROTOLIB_PATH=/usr/local/Ascend/runtime/ops/op_proto/built-n
export ASCEND_OPP_PATH=/usr/local/Ascend/ops

export BERT_BASE_DIR=chinese_L-12_H-768_A-12
export GLUE_DIR=chineseGLUEdatasets.v0.0.1

TASK_NAME="msraner"

python3 run_ner.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=${TASK_NAME}_output/ |& tee train.log
