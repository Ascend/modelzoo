source /path/to/modelzoo_resnet50/scripts/env.sh
if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [NODE_NUM]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
if [ $# == 3 ]
then
        echo "the node num : $3"
    #PATH3=$(get_real_path $3)
fi

if [ ! -f "$PATH1" ]
then
    echo "error: MINDSPORE_HCCL_CONFIG_PATH=$PATH1 is not a file"
exit 1
fi

if [ ! -d "$PATH2" ]
then
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=32
export MINDSPORE_HCCL_CONFIG_PATH=$PATH1
export RANK_TABLE_FILE=$PATH1
NUM_NODE=$3
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    let rank_id=i+NUM_NODE*8
    export RANK_ID=$rank_id
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cd ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python /path/to/modelzoo_resnet50/train.py --per_batch_size=256 --data_dir=$PATH2 --is_distributed=1 --lr_scheduler=cosine_annealing --label_smooth=1 --T_max=90 --max_epoch=90 --backbone=resnet50 --lr=3.2 --warmup_epochs=5 --ckpt_interval=1560 &> log &
    cd ..
done