#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/
labels=$cur_dir/../labels/

if [ ! -d "$log_dir" ];then
	mkdir -p $log_dir
fi

if [ ! -d "$dataset" ];then
	mkdir -p $dataset
fi

if [ ! -d "$labels" ];then
	mkdir -p $labels
fi

cd $cur_dir
python3 preprocess.py ../ori_images/tfrecord/ $dataset ../labels/



cd $cur_dir/../Benchmark/out/
chmod +x benchmark

echo "./benchmark --dataDir $dataset --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp"
./benchmark --dataDir $dataset --om ../../model/unet3d_1batch.om --batchSize 1 --modelType unet3d --imgType yuv --deviceId 0 --framework tensorflow --useDvpp 0


cd $cur_dir/

python3 $cur_dir/postprocess.py $cur_dir/../results/unet3d/ $cur_dir/../labels/

