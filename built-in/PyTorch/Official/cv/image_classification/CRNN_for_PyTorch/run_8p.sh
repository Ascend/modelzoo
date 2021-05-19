#!/usr/bin/env bash
source npu_set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export SWITCH_MM_OUTPUT_ENABLE=1

/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 4

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

for i in $(seq 0 7)
do
python3.7 ${currentDir}/main_8p.py --cfg ${currentDir}/LMDB_8p_config.yaml --npu ${i} > ./crnn_8p_${i}_npu.log 2>&1 &
done
