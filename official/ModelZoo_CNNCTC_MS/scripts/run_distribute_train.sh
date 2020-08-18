#!/bin/bash
current_exec_path=$(pwd)
echo ${current_exec_path}

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)

python ${current_exec_path}/scripts/generate_hccn_file.py
export RANK_TABLE_FILE=${current_exec_path}/scripts/rank_table_8p.json
export RANK_SIZE=8
ulimit -u unlimited
for((i=0;i<=7;i++));
do
    rm ./train_parallel_device_$i/ -rf
    #rm ge_* -rf
    mkdir ./train_parallel_device_$i
    cp ./*.py ./train_parallel_device_$i
    cp ./scripts/*.sh ./train_parallel_device_$i
    cp -r ./src ./train_parallel_device_$i
    cd ./train_parallel_device_$i
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    if [ -f $PATH1 ]
    then
      python train_para.py --device_id $i --ckpt_path=$PATH1 >log_device_$i.log 2>&1 &
    else
      python train_para.py --device_id $i >log_device_$i.log 2>&1 &
    fi
    cd ..
done

