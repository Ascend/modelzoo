export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error

cur_path=`pwd`

export RANK_SIZE=4
export RANK_TABLE_FILE=$cur_path/configs/${RANK_SIZE}p.json

RANK_ID_START=0

BASE_PATH=`pwd`
cd $BASE_PATH/models/research
LOG_PATH=log

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
  TMP_PATH=$LOG_PATH/D$RANK_ID
  mkdir -p $TMP_PATH
  nohup bash examples/SSD320_FP16_4GPU.sh /checkpoints/ ./configs/ $RANK_ID  > $TMP_PATH/train_$RANK_ID.log 2>&1 &
done
