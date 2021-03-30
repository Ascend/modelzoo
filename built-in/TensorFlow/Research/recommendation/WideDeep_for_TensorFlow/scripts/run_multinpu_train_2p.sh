#!/bin/bash
current_exec_path=$(pwd)
echo ${current_exec_path}

echo ${self_path}

export RANK_TABLE_FILE=${current_exec_path}/rank_table_2p.json
#export RANK_TABLE_FILE=${current_exec_path}/rank_table_2p_114.json
#export MINDSPORE_HCCL_CONFIG_PATH=${current_exec_path}/rank_table_8p.json
export RANK_SIZE=2
#export ME_DRAW_GRAPH=1


for((i=0;i<=1;i++));
do
    rm ${current_exec_path}/device_$i/ -rf
    #rm ge_* -rf
    mkdir ${current_exec_path}/device_$i
    cd ${current_exec_path}/device_$i
    export RANK_ID=$i
    export DEVICE_ID=$i
    #cd ${current_exec_path}
    python -u ${current_exec_path}/tools/train_multinpu.py  >test_deep$i.log 2>&1 &
done



