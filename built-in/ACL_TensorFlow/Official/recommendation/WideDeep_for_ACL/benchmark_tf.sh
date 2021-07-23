#!/bin/bash

for para in $*
do
    if [[ $para == --batchSize* ]];then
        batchSize=`echo ${para#*=}`
    elif [[ $para == --dataPath* ]];then
        dataPath=`echo ${para#*=}`
    elif [[ $para == --modelPath* ]];then
        modelPath=`echo ${para#*=}`
    fi
done

rm -rf *.txt
rm -rf *.pbtxt
rm -rf dump*
rm -rf kernel_meta
rm -rf result*

CUR_DIR=$(dirname $(readlink -f $0))
echo $CUR_DIR
echo '-------cur dir---------'

$CUR_DIR/benchmark widedeep $batchSize 0 $modelPath $dataPath

echo "-----------------Accuracy Summary------------------"
cat $CUR_DIR/widedeep_outputfile.txt
echo "----------------------------------------------------"