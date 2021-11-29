#!/bin/bash

currentDir=$(cd "$(dirname "$0")";pwd)
export SET_RESULT_FILE=$currentDir/set_result.py
export RESULT_FILE=$currentDir/result.txt

function prepare() {
    # download dataset

    # verify  dataset

    # preprocess
    return 0
}

function exec_train() {

    # pytorch lenet5 sample
    #python3.7 $currentDir/pytorch_lenet5_train.py

    # tensorflow-1.15 wide&deep sample
    #python3.7 $currentDir/tensorflow_1_15_wide_deep.py

    # test sample
    cd $currentDir/test
    bash train_full_8p.sh
    FPS=`grep "hooks.py:141"  $currentDir/test/output/1/train_1.log |awk -F "," 'END {print $4}' |awk -F ":" '{print $2}' | awk -F "." '{print $1}'`
    train_accuracy=`grep "accuracy =" $currentDir/test/output/1/train_1.log|awk -F "," 'END {print $1}'|awk -F " " 'END {print $NF}'`

    python3.7 $currentDir/set_result.py training "accuracy" $train_accuracy
    python3.7 $currentDir/set_result.py training "throughput_ratio" $FPS
    python3.7 $currentDir/set_result.py training "result" "NOK"
}

function main() {

    prepare

    exec_train

}

main "$@"
ret=$?
exit $?
