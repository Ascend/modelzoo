#prepare
steps=10
pid=`pgrep nmt | head -n 1`
if [ -n "$pid" ];then
  kill -9 $(pgrep nmt)
  sleep 15
fi
CUR_DIR=$(dirname $(readlink -f $0))
DATA_DIR=$CUR_DIR/nmt_data
OUTPUT_DIR1=/tmp/nmt_output1
OUTPUT_DIR2=/tmp/nmt_output2
OUTPUT_DIR3=/tmp/nmt_output3
OUTPUT_DIR4=/tmp/nmt_output4
OUTPUT_DIR5=/tmp/nmt_output5
OUTPUT_DIR6=/tmp/nmt_output6
OUTPUT_DIR7=/tmp/nmt_output7
OUTPUT_DIR8=/tmp/nmt_output8
export RANK_SIZE=1
export JOB_ID=123456
export USE_NPU=True
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export DDK_ENV_FLAG=1
#export TF_CPP_MIN_LOG_LEVEL=3
#export DDK_VERSION_FLAG=1.60.T41.0.B182
#export PATH=$PATH:/usr/local/HiAI/runtime/ccec_compiler/bin
#export PYTHONPATH=$PYTHONPATH:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:$CUR_DIR
#export CUSTOM_OP_LIB_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
#export LD_LIBRARY_PATH=/usr/local/HiAI/runtime/lib64:/usr/local/HiAI/driver/lib64/:/usr/local/lib:/usr/lib/:/usr/local/python3.7/lib/
export DDK_VERSION_FLAG=1.60.T17.B830
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:$CUR_DIR
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
#cp -rf conf/8.json /usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json
su - HwHiAiUser -c "adc --host 127.0.0.1:22118 --log \"SetLogLevel(0)[ERROR]\""
su - HwHiAiUser -c "adc --host 127.0.0.1:22118 --log \"SetLogLevel(0)[ERROR]\" --device 4"
rm -fr $OUTPUT_DIR1
rm -fr $OUTPUT_DIR2
rm -fr $OUTPUT_DIR3
rm -fr $OUTPUT_DIR4
rm -fr $OUTPUT_DIR5
rm -fr $OUTPUT_DIR6
rm -fr $OUTPUT_DIR7
rm -fr $OUTPUT_DIR8
mkdir -p $OUTPUT_DIR1
mkdir -p $OUTPUT_DIR2
mkdir -p $OUTPUT_DIR3
mkdir -p $OUTPUT_DIR4
mkdir -p $OUTPUT_DIR5
mkdir -p $OUTPUT_DIR6
mkdir -p $OUTPUT_DIR7
mkdir -p $OUTPUT_DIR8

#execute
cd $OUTPUT_DIR1
DEVICE_ID=0 RANK_ID=0 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR1 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_1.log 2>&1 &
cd $OUTPUT_DIR2
DEVICE_ID=1 RANK_ID=1 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR2 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_2.log 2>&1 &
cd $OUTPUT_DIR3
DEVICE_ID=2 RANK_ID=2 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR3 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_3.log 2>&1 &
cd $OUTPUT_DIR4
DEVICE_ID=3 RANK_ID=3 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR4 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_4.log 2>&1 &
cd $OUTPUT_DIR5
DEVICE_ID=4 RANK_ID=4 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR5 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_5.log 2>&1 &
cd $OUTPUT_DIR6
DEVICE_ID=5 RANK_ID=5 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR6 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_6.log 2>&1 &
cd $OUTPUT_DIR7
DEVICE_ID=6 RANK_ID=6 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR7 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_7.log 2>&1 &
cd $OUTPUT_DIR8
DEVICE_ID=7 RANK_ID=7 python3 -m nmt.nmt --attention=scaled_luong --src=vi --tgt=en --num_gpus=1  --vocab_prefix=$DATA_DIR/vocab  --train_prefix=$DATA_DIR/train  --dev_prefix=$DATA_DIR/tst2012  --test_prefix=$DATA_DIR/tst2013  --out_dir=$OUTPUT_DIR8 --num_train_steps=$steps  --steps_per_stats=100 --num_layers=2 --num_units=128  --dropout=0.2 --batch_size=128 --metrics=bleu > $CUR_DIR/nmt_train_8_8.log 2>&1 &
