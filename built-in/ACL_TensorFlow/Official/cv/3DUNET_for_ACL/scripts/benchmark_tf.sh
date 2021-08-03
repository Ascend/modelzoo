#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/
labels=$cur_dir/../labels/

if [ ! -d "$log_dir" ];then
	mkdir -p $log_dir
fi

preprocess()
{
    if [ ! -d "$dataset" ];then
	    mkdir -p $dataset
    fi
    if [ ! -d "$labels" ];then
	    mkdir -p $labels
    fi
    cd $cur_dir
    python3 preprocess.py ../ori_images/tfrecord/ $dataset ../labels/
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
    echo "start parse 3dunet" >> $testcase_dir/run.log
    python3 $cur_dir/postprocess.py $cur_dir/../results/${modelType}/ $cur_dir/../labels/ >> $testcase_dir/run.log 2>&1


    TumorCore=`cat $testcase_dir/run.log | grep 'Total result: TumorCore' | awk '{print $4}'`
    PeritumoralEdema=`cat $testcase_dir/run.log | grep 'PeritumoralEdema:' | awk '{print $6}'`
    EnhancingTumor=`cat $testcase_dir/run.log | grep 'EnhancingTumor:' | awk '{print $8}'`
    MeanDice=`cat $testcase_dir/run.log | grep 'MeanDice:' | awk '{print $10}'`
    WholeTumor=`cat $testcase_dir/run.log | grep 'WholeTumor' | awk '{print $12}'`

    echo TumorCore: $TumorCore>> $testcase_dir/precision_results.log
    echo PeritumoralEdema: $PeritumoralEdema>> $testcase_dir/precision_results.log
    echo EnhancingTumor: $EnhancingTumor>> $testcase_dir/precision_results.log
    echo MeanDice: $MeanDice>> $testcase_dir/precision_results.log
    echo WholeTumor: $WholeTumor>> $testcase_dir/precision_results.log
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
    --modelType           model type(3Dunet)
    --imgType             input image format type(rgb/yuv/raw/,default is yuv)
    --precision           precision type(fp16/fp32/int8,default is fp16)
    --inputType           input data type(fp32/fp16/int8,default is fp32)
    --outType             inference output type(fp32/fp16/int8,default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp,default is 0)
    --deviceId            running device ID(default is 0)
    --framework            (caffe,MindSpore,tensorflow,default is tensorflow)
    --modelPath           (../../model/3dunet.om)
    --dataPath            (../../datasets/)
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/3dunet.om --dataPath=../../datasets/ --modelType=3dunet --imgType=rgb
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

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%s")
mkdir -p $testcase_dir

echo "======================preprocess test======================"
preprocess
echo "======================infer test======================"
infer_test
echo "======================collect test======================"
collect_result
echo "======================end======================"


