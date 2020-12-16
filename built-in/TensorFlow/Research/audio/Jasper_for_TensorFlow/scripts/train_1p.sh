#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/1p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/1p/${device_id}
#rm -rf ${currentDir}/result/1p/${device_id}/*
cd ${currentDir}/result/1p/${device_id}

#start exec
source /root/archiconda3/bin/activate ci3.7
if [ -d "${currentDir}/result/1p/${device_id}/jasper_log_folder/logs" ]
then
    python3 ${currentDir}/../run.py --config_file=${currentDir}/../configs/speech2text/jasper5x3_LibriSpeech_nvgrad_masks_1p.py --mode=train --continue_learning --enable_logs 2>&1 | tee -a  ${currentDir}/result/1p/train_${device_id}.log
else
    python3 ${currentDir}/../run.py --config_file=${currentDir}/../configs/speech2text/jasper5x3_LibriSpeech_nvgrad_masks_1p.py --mode=train --enable_logs 2>&1 | tee ${currentDir}/result/1p/train_${device_id}.log
fi

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/1p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/1p/train_${device_id}.log
fi

