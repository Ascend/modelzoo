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
    cd $currentDir/test/
    bash train_full_p.sh --data_path=/data/GRU4Rec_for_TensorFlow/rsc15
    
    sleep 5
    accuracy="81"
    FPS=`grep TimeHistory  $currentDir/test/output/1/train.log|awk 'END {print $2}'`
    ActualLoss=`awk 'END {print}' $currentDir/test/output/1/train_GRU4Rec_for_TensorFlow_bs50_1p_acc_loss.txt`

    python3.7 $currentDir/set_result.py training "accuracy" $Actualoss
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
