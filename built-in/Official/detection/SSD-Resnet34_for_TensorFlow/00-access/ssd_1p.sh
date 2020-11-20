export RANK_SIZE=1
RANK_ID_START=1

SAVE_PATH=training
BASE_PATH=`pwd`
echo $BASE_PATH

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device "$RANK_ID
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp exec_main.sh $TMP_PATH/
cd $TMP_PATH
bash exec_main.sh $RANK_ID $RANK_SIZE $BASE_PATH > train_$RANK_ID.log &
cd -
done






