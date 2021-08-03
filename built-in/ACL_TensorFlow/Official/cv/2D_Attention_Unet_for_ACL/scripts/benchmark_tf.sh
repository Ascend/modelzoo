#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/

if [ ! -d "$log_dir" ];then
	mkdir -p $log_dir
fi

if [ -f "$cur_dir/data.txt" ];then
	rm $cui_dir/data.txt
fi

preprocess()
{
    if [ ! -d "$dataset" ];then
	    mkdir -p $dataset
    fi
    cd $cur_dir
    python3 processdata_test.py --dataset=../image_ori/lashan --output_path=$dataset --crop_width=224 --crop_height=224
}

infer_test()
{
    cd $cur_dir/../Benchmark/out/
    chmod +x benchmark
    if [[ $? -ne 0 ]];then
	    echo "benchmark function execute failed"
	    exit 1
    fi

    echo "./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp"
   ./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp >performance.log 
}

collect_result()
{
    cp $cur_dir/../Benchmark/out/test_perform_static.txt $testcase_dir/data.log
    AiModel_time=`cat $testcase_dir/data.log | tail -1 | awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    AiModel_throughput=`cat $testcase_dir/data.log | tail -1 |awk '{print $8}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    InferenceEngine_total_time=`cat $testcase_dir/data.log | awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    echo "InferencePerformance: $AiModel_time ms/batch,$AiModel_throughput images/sec" >>$testcase_dir/performance_results.log
    echo "InferenceTotalTime: $InferenceEngine_total_time ms" >>$testcase_dir/performance_results.log

    echo "start parse 2DAttention_Unet" >> $testcase_dir/run.log

    python3 $cur_dir/afterprocessdata_test.py --path=../../results/${modelType}/ --dataset=../../image_ori/lashan --benchmark_path=$trueValuePath >> $testcase_dir/run.log 2>&1

    accuracy=`cat $testcase_dir/run.log | grep 'Average test accuracy' | awk '{print $5}'`
    Road=`cat $testcase_dir/run.log | grep 'Road' | awk '{print $3}'`
    Others=`cat $testcase_dir/run.log | grep 'Others' | awk '{print $3}'`
    precision=`cat $testcase_dir/run.log | grep 'Average precision' | awk '{print $4}'`
    F1_score=`cat $testcase_dir/run.log | grep 'Average F1 score' | awk '{print $5}'`
    IoU=`cat $testcase_dir/run.log | grep 'Average mean IoU score' | awk '{print $6}'`

    echo accuracy: $accuracy >> $testcase_dir/precision_results.log
    echo Road: $Road >> $testcase_dir/precision_results.log
    echo Others: $Others >> $testcase_dir/precision_results.log
    echo precision: $precision >> $testcase_dir/precision_results.log
    echo F1_score: $F1_score >> $testcase_dir/precision_results.log
    echo IoU: $IoU >> $testcase_dir/precision_results.log
    #print results
    echo ""
    cat $testcase_dir/performance_results.log | grep InferencePerformance
    cat $testcase_dir/precision_results.log | tail -6
}
########################################################

if [[ $1 == --help || $1 == -h || $# -eq 0 ]];then
    echo "usage: ./benchmark_tf.sh <args>"
echo ""
    echo 'parameter explain:
    --batchSize           Data number for one inference
    --modelType           model type(2DAttention_Unet)
    --imgType             input image format type(rgb/yuv/raw/,default is yuv)
    --precision           precision type(fp16/fp32/int8,default is fp16)
    --inputType           input data type(fp32/fp16/int8,default is fp32)
    --outType             inference output type(fp32/fp16/int8,default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp,default is 0)
    --deviceId            running device ID(default is 0)
    --framework            (caffe,MindSpore,tensorflow,default is tensorflow)
    --modelPath           (../../model/2DAttention_fp16_1batch.om)
    --dataPath            (../../datasets/)
    --trueValuePath       the path of true Value, default is ../../image_ori/Val
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/2DAttention_fp16_1batch.om --dataPath=../../datasets/ --modelType=2DAttention_Unet --imgType=rgb --trueValuePath=../../image_ori/Val/
#    '
   exit 1
fi
loopNum=1
deviceId=0
imgType=yuv
precision=fp16
inputType=fp32
outputType=fp32
useDvpp=0
framework=tensorflow
trueValuePath=../../image_ori/Val/
for para in $*
do
    if [[ $para == --batchSize* ]];then
	batchSize=`echo ${para#*=}`
    elif [[ $para == --modelType* ]];then
	modelType=`echo ${para#*=}`
    elif [[ $para == --imgType* ]];then
	imgType=`echo ${para#*=}`
    elif [[ $para == --loopNum* ]];then
	loopNum=`echo ${para#*=}`
    elif [[ $para == --deviceId* ]];then
	deviceId=`echo ${para#*=}`
    elif [[ $para == --precision* ]];then
	precision=`echo ${para#*=}`
    elif [[ $para == --outType* ]];then
	outputType=`echo ${para#*=}`
    elif [[ $para == --useDvpp* ]];then
	useDvpp=`echo ${para#*=}`
    elif [[ $para == --shape* ]];then
	shape=`echo ${para#*=}`
    elif [[ $para == --inputType* ]];then
	inputType=`echo ${para#*=}`
    elif [[ $para == --framework* ]];then
	framework=`echo ${para#*=}`
    elif [[ $para == --dataPath* ]];then
	dataPath=`echo ${para#*=}`
    elif [[ $para == --modelPath* ]];then
	modelPath=`echo ${para#*=}`
    elif [[ $para == --cmdType* ]];then
	cmdType=`echo ${para#*=}`
    elif [[ $para == --trueValuePath* ]];then
	trueValuePath=`echo ${para#*=}`
    fi
done

rm -rf $cur_dir/../results
rm -rf $cur_dir/file.log
rm -rf $log_dir
rm -rf $cur_dir/../Benchmark/out/result*
rm -rf $cur_dir/../Benchmark/out/performance.log

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%S")
mkdir -p $testcase_dir

echo "======================preprocess test======================"
preprocess
echo "======================infer test======================"
infer_test
echo "======================collect test======================"
collect_result
echo "======================end======================"


