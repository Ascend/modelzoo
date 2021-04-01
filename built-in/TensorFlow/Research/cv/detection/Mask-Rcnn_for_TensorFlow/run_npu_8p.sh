#clean slog
#rm -rf /var/log/npu/slog/host-0/*.log
#rm -rf /var/log/npu/slog/device-*/*.log
#rm -rf /var/log/npu/slog/
BASE_PATH=$(cd "$(dirname "$0")"; pwd)
# set env
export PYTHONPATH=/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu:/usr/local/python3.7.5/lib/
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin/:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=1
#unset DUMP_GE_GRAPH
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1
#export PRINT_MODEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#export DUMP_OP=1
export HCCL_CONNECT_TIMEOUT=6000

export EXPERIMENTAL_DYNAMIC_PARTITION=1

export RANK_TABLE_FILE=$BASE_PATH/npu_config/8p.json

# for fast training
#unset DUMP_OP
#unset PRINT_MODEL
#unset DUMP_GE_GRAPH
export DISABLE_REUSE_MEMORY=0
export TF_CPP_MIN_LOG_LEVEL=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export RANK_SIZE=8
RANK_ID_START=0

SAVE_PATH=training

echo $BASE_PATH


for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo
#su HwHiAiUser -c "/usr/local/Ascend/toolkit/bin/adc --host 127.0.0.1:22118 --log \"SetLogLevel(0)[error]\" --device "$RANK_ID
/usr/local/Ascend/driver/tools/msnpureport -d $RANK_ID -g error
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp exec_main.sh $TMP_PATH/
#cp profSetEnvForNoDocker.sh $TMP_PATH/
cd $TMP_PATH
# profiling
#source profSetEnvForNoDocker.sh $RANK_ID
# run
bash exec_main.sh $RANK_ID $RANK_SIZE $BASE_PATH 2>&1 | tee train_$RANK_ID.log &
cd -
done






