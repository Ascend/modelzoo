#!/bin/bash

function fn_ssh_cmd()
{
        ip=$1
        usr=$2
        host_passwd=$3
        ssh_command=$4
/usr/bin/expect <<EOF
spawn ssh $usr@$ip
set timeout 5
expect {
"*yes/no*" { send "yes\r"; exp_continue }
"*assword:" { send "$host_passwd\r" }
}
expect "#"
send "$ssh_command\r"
expect "#"
send "exit\r"
expect eof
EOF
}


echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Change device devmem 0xaf10A000 64 to 0x0000000000000004 to open PMU."
su HwHiAiUser -c "adc --host 127.0.0.1:22118 --log \"SetLogLevel(0)[error]\""
su HwHiAiUser -c "adc --host 127.0.0.1:22118 --log \"SetLogLevel(0)[error]\" --device 4"


ifconfig endvnic 192.168.1.111 netmask 255.255.255.0
echo > /root/.ssh/known_hosts
fn_ssh_cmd 192.168.1.199 HwHiAiUser "Huawei2012#" "echo -e 'Huawei12#$' | su -c '/sbin/devmem 0xaf10A000 64 0x0000000000000004' root" >/dev/null 2>&1

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Config /etc/profile.cfg to turn on profiling mode."
echo "cp -rf ${current_path}/profile.cfg /var/log/npu/conf/profiling/profile.cfg"
mv /var/log/npu/conf/profiling/profile.cfg /var/log/npu/conf/profiling/profile.cfg.bak
cp -rf profile.cfg /var/log/npu/conf/profiling/profile.cfg
chown HwHiAiUser:HwHiAiUser /var/log/npu/conf/profiling/profile.cfg
chmod 777 /var/log/npu/conf/profiling/profile.cfg
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Restart IDE-daemon-host"
#pkill IDE-daemon-host
#sleep 20
#su HwHiAiUser -c "/usr/local/HiAI/driver/tools/IDE-daemon-host &"


currentDir=$(cd "$(dirname "$0")"; pwd)
#export PYTHONPATH=/usr/local/python3.7/lib/python3.7/site-packages:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe/:${currentDir}:../../
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe/:../../../
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/HiAI/fwkacllib/lib64/:/usr/local/HiAI/driver/lib64/common/:/usr/local/HiAI/driver/lib64/driver/:/usr/local/HiAI/add-ons/:/usr/lib/x86_64-linux-gnu
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/HiAI/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/HiAI/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export DUMP_GE_GRAPH=1
export DUMP_GRAPH_LEVEL=3
#export DUMP_OP=1
#export SLOG_PRINT_TO_STDOUT=1
#export DISABLE_REUSE_MEMORY=1
export JOB_ID=10086
export DEVICE_ID=0
export DEVICE_INDEX=0
#export PRINT_MODEL=1
export RANK_ID=0
export RANK_SIZE=8
export RANK_TABLE_FILE=../configs/8p.json
export FUSION_TENSOR_SIZE=1000000000

source open_profiling.sh

#sleep 5
rm -rf model_ckpt/*
rm -rf result/exec-0000/npu/*

ulimit -c 0
python3.7 -u ncf_estimator_main.py \
      --model_dir './model_ckpt' \
      --data_dir '../movielens_data' \
      --dataset ml-20m \
      --train_epochs 1 \
      --batch_size 1048576 \
      --eval_batch_size 160000 \
      --learning_rate 0.00382059 \
      --layers 64,32,16 \
      --num_factors 16 \
      --hr_threshold 1.0 \
      --accelerator '1980' \
      --print_freq 1 \
      --gpu_num 8 \
      --debug_mod \
      --debug_steps 10 \
      #--seed 1 \
      #--use_synthetic_data \

#cp -r /var/log/npu/slog /var/log/npu/profiling  result/exec-0000/npu
#python3.7 performanceanalysis.py 

