#!/bin/bash
#set -x
cur_dir=`pwd`
dataset=$cur_dir/../datasets/
loopNum=1
deviceId=0
imgType=yuv
precision=fp16
inputType=fp32
outputType=fp32
useDvpp=0
framework=tensorflow
trueValuePath=../../image_ori/Val/

if [ ! -d "$dataset" ];then
	mkdir -p $dataset
fi
cd $cur_dir

python3 processdata_test.py --dataset=../image_ori/lashan --output_path=$dataset --crop_width=224 --crop_height=224

../Benchmark/out/benchmark --dataDir $dataset --om ../model/2DAttention_fp16_1batch.om --batchSize 1 --modelType 2DAttention_unet --imgType yuv --deviceId 0 --framework tensorflow --useDvpp 0


python3 $cur_dir/afterprocessdata_test.py --path=../../results/2DAttention_unet/ --dataset=../image_ori/lashan --benchmark_path=../image_ori/Val/
