
#clean slog
rm -rf /var/log/npu/slog/host-0/*.log
rm -rf /var/log/npu/slog/device-*/*.log

# setting main path
MAIN_PATH=$(dirname $(readlink -f $0))

# set env
export PYTHONPATH=/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/:$MAIN_PATH/../../../
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T49.0.B201
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export DUMP_GE_GRAPH=1
export DUMP_GRAPH_LEVEL=3
export PRINT_MODEL=1
export SLOG_PRINT_TO_STDOUT=1

ulimit -c unlimited

export RANK_SIZE=$1
export RANK_TABLE_FILE=./configs/${RANK_SIZE}p.json
RANK_ID_START=0


SAVE_PATH=training/t1
rm -rf $SAVE_PATH
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device "$RANK_ID
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp run_test.sh $TMP_PATH/
cp $RANK_TABLE_FILE $TMP_PATH/rank_table.json
cd $TMP_PATH
nohup bash run_test.sh $RANK_ID $RANK_SIZE $MAIN_PATH > train_$RANK_ID.log &
cd -
done

wait
rm perf.log
cat nohup.out | egrep "INFO:tensorflow:loss|INFO:tensorflow:Benchmark metric: {'name': 'current_examples_per_sec', ">perf.log


