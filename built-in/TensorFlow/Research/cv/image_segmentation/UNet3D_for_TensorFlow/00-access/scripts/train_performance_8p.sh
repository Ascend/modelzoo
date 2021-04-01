#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

data_dir=$1
fold=$2


upDir=$(dirname "$PWD")
out_dir=${upDir}/output
fold_dir=${out_dir}/fold-$fold/
work_dir=${fold_dir}/Device-${ASCEND_DEVICE_ID}
mkdir -p ${work_dir}

rm -rf ${fold_dir}/fold-${fold}_Device-${ASCEND_DEVICE_ID}.log
rm -rf ${work_dir}/*

cd ${work_dir}
python3 ${upDir}/main_npu.py --data_dir=${data_dir} --model_dir=result --exec_mode=train --npu_loss_scale=1048576 --max_steps=640 --benchmark --fold=${fold} --batch_size=2 --augment   > ${fold_dir}/fold-${fold}_Device-${ASCEND_DEVICE_ID}.log 2>&1

if [ x"${ASCEND_DEVICE_ID}" = x0 ] ;
then
    result=`cat ${fold_dir}/fold-${fold}_Device-${ASCEND_DEVICE_ID}.log | grep DLL`
    echo "$result"
fi
