
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
export ASCEND_GLOBAL_LOG_LEVEL=3

# local variable
RANK_SIZE=1
#RANK_TABLE_FILE=./hccl_config/${RANK_SIZE}p.json
RANK_ID_START=0
SAVE_PATH=training/t1

# training stage
MODE=single # optional multi

# hyperparameter
data_path=''
save_dir='./training/'
batch_size=32

if [[ $1 == --help || $1 == -h ]];then
    echo "usage:./npu_train.sh <args>"
    echo " "
    echo "parameters:
    --data_path            path to train annotation file, must be specified
    --save_dir             path to save ckpt
    --batch_size           batchsize, default is 16
    --mode                 single or multi scale train, default is single
    --rank_size            num of NPUs for training, default is 1"
fi

# params override
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --save_dir* ]];then
        save_dir=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --mode* ]];then
        MODE=`echo ${para#*=}`
    elif [[ $para == --rank_size* ]];then
        RANK_SIZE=`echo ${para#*=}`
    fi
done

if [[ $data_path == '' ]];then
    echo "[ERROR] para \"data_path\" must be specified"
    exit 1
fi

if [ $RANK_SIZE -gt 1 ];then
export RANK_TABLE_FILE=./hccl_config/${RANK_SIZE}p.json
fi

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo

TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cp run_yolov3.sh $TMP_PATH/
cd $TMP_PATH
nohup bash run_yolov3.sh --RANK_ID=$RANK_ID --RANK_SIZE=$RANK_SIZE --MAIN_PATH=$MAIN_PATH --MODE=$MODE --data_path=$data_path --save_dir=$save_dir --batch_size=$batch_size > train_$RANK_ID.log &
cd -

done
