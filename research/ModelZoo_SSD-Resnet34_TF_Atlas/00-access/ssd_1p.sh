#clean slog
rm -rf /var/log/npu/slog/host-0/*.log
rm -rf /var/log/npu/slog/device-*/*.log

# set env
export PYTHONPATH=/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/:../:/opt/npu/x00558890/item_npu3/thirdlib:/opt/npu/x00558890/item_npu3/tpu-v3-32-ssd
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export DUMP_GE_GRAPH=1
export DUMP_GRAPH_LEVEL=3
export PRINT_MODEL=1
export SLOG_PRINT_TO_STDOUT=1


export RANK_SIZE=1
RANK_ID_START=1

SAVE_PATH=training
BASE_PATH=`pwd`
echo $BASE_PATH

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo
su HwHiAiUser -c "adc --host 10.246.246.77:22118 --log \"SetLogLevel(0)[error]\" --device "$RANK_ID
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp exec_main.sh $TMP_PATH/
cd $TMP_PATH
bash exec_main.sh $RANK_ID $RANK_SIZE $BASE_PATH > train_$RANK_ID.log &
cd -
done






