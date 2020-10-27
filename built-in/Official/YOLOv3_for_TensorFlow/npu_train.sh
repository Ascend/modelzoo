
#clean slog
#rm -rf /var/log/npu/slog/host-0/*.log
#rm -rf /var/log/npu/slog/device-*/*.log
#rm -rf aicpu*
#rm -rf ge_proto*
#rm -rf *.pbtxt
#rm -rf training

# setting main path
MAIN_PATH=$(dirname $(readlink -f $0))

# set env


# local variable
RANK_SIZE=$1
RANK_TABLE_FILE=./hccl_config/${RANK_SIZE}p.json
RANK_ID_START=0
SAVE_PATH=training/t1

# training stage
MODE=$2

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device "$RANK_ID
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp run_yolov3.sh $TMP_PATH/
cp $RANK_TABLE_FILE $TMP_PATH/rank_table.json
cd $TMP_PATH
nohup bash run_yolov3.sh $RANK_ID $RANK_SIZE $MAIN_PATH $MODE > train_$RANK_ID.log &
cd -

done




