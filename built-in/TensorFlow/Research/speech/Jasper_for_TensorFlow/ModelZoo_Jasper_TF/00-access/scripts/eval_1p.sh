#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")

logDir=${currentDir}/result/8p
[ -d "${currentDir}/result/1p" ] && logDir=${currentDir}/result/1p
logFile=${logDir}/eval_${device_id}.log

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing eval fail" >> ${logFile}
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}




cd ${logDir}/${device_id}

#start exec
source /root/archiconda3/bin/activate ci3.7
if [ -d "${logDir}/${device_id}/jasper_log_folder/logs" ]
then
    python ${currentDir}/../run.py --config_file=${currentDir}/../configs/speech2text/jasper5x3_LibriSpeech_nvgrad_masks_1p.py --mode=eval --enable_logs 2>&1 | tee ${logFile}
    #python ${currentDir}/../run.py --config_file=${currentDir}/../configs/speech2text/jasper5x3_LibriSpeech_nvgrad_masks_1p.py --mode=infer --enable_logs 2>&1 | tee ${logFile}
fi

if [ $? -eq 0 ] ;
then
    echo "turing eval success" >> ${logFile}
else
    echo "turing eval fail" >> ${logFile}
fi

