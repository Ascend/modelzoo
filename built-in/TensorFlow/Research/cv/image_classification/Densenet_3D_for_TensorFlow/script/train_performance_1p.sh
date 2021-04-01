#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

data_dir=$1

upDir=$(dirname "$PWD")
out_dir=${upDir}/output
work_dir=${out_dir}/Device-${ASCEND_DEVICE_ID}
mkdir -p ${work_dir}

#rm -rf ${work_dir}/*

cd ${work_dir}

python3.7 ${upDir}/train.py -bs 2 -gpu 0 -mn dense24 -sp dense24_correction -nc True -e 5 -r ${data_dir}  > Device-${ASCEND_DEVICE_ID}.log 2>&1

#python3.7 ${upDir}/test.py -gpu 0 -m dense24_correction-4 -mn dense24 -nc True -r ${data_dir} > test-${ASCEND_DEVICE_ID}.log 2>&1
