#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
datasets1=$cur_dir/../datasets/input_flair/
datasets2=$cur_dir/../datasets/input_t1/

if [ ! -d "$log_dir" ];then
	mkdir -p $log_dir
fi

preprocess()
{
    if [ ! -d "$datasets1" ];then
	    mkdir -p $datasets1
    fi
    if [ ! -d "$datasets2" ];then
	    mkdir -p $datasets2
    fi
    cd $cur_dir 
    python3 preprocess.py -m ../ori_images/npu/dense24_correction-4 -mn dense24 -nc True -r ../ori_images/BRATS2017/Brats17ValidationData/ -input1 $datasets1 -input2 $datasets2
    echo "preprocess succ."
}

infer_test()
{
    cd $cur_dir/../Benchmark/out/
    chmod +x benchmark
    if [[ $? -ne 0 ]];then
	    echo "benchmark function execute failed."
	    exit 1
    fi

    echo "./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchsize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --usrDvpp $usrDvpp"
   ./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchsize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --usrDvpp $usrDvpp >performance.log 
}

collect_result()
{
    cp $cur_dir/../Benchmark/out/test_perform_static.txt $testcase_dir/data.log
    AiModel_time=`cat $testcase_dir/data.log | tail -1 | awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    AiModel_throughput=`cat $testcase_dir/data.log | tail -1 |awk '{print $8}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    InferenceEngine_total_time=`cat $testcase_dir/data.log | awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    echo "InferencePerformance: $AiModel_time ms/batch,$AiModel_throughput images/sec" >>$testcase_dir/performance_results.log
    echo "InferenceTotalTime: $InferenceEngine_total_time ms" >>$testcase_dir/performance_results.log
    echo "start parse densenet24" >> $testcase_dir/run.log
    python3 $cur_dir/postprocess.py -m $cur_dir/../ori_images/npu/dense24_correction-4 -mn dense24 -nc True -r $cur_dir/../ori_images/BRATS2017/Brats17ValidationData/ -rp $cur_dir/../results/$modelType/ >> $testcase_dir/run.log 2>&1


    Whole=`cat $testcase_dir/run.log | grep -A1 'mean dice whole:' | grep -v 'mean' | awk '{print $1}' | awk -F '[' '{print $2}'`
    PeritumoralEdema=`cat $testcase_dir/run.log | grep -A1 'mean dice core:' | grep -v 'mean' | awk '{print $1}' | awk -F '[' '{print $2}'`
    Tumor=`cat $testcase_dir/run.log | grep -A1 'mean dice enhance:' | grep -v 'mean' | awk '{print $1}' | awk -F '[' '{print $2}'`

    echo TumorCore: $Whole>> $testcase_dir/precision_results.log
    echo PeritumoralEdema: $PeritumoralEdema>> $testcase_dir/precision_results.log
    echo EnhancingTumor: $Tumor>> $testcase_dir/precision_results.log
    #print results
    echo ""
    cat $testcase_dir/performance_results.log | grep InferencePerformance
    cat $testcase_dir/precision_results.log | tail -6
}
########################START################################

if [[ $1 == --help || $1 == -h || $# -eq 0 ]];then
    echo "usage: ./benchmark_tf.sh <args>"
echo ""
    echo 'parameter explain:
    --batchSize           Data number for one inference
    --modelType           model type(densenet24)
    --imgType             input image format type(rgb/yuv/raw/,default is rgb)
    --precision           precision type(fp16/fp32/int8,default is fp16)
    --inputType           input data type(fp32/fp16/int8,default is fp32)
    --outType             inference output type(fp32/fp16/int8,default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp,default is 0)
    --deviceId            running device ID(default is 0)
    --framework            (caffe,MindSpore,tensorflow,default is tensorflow)
    --modelPath           (../../model/densenet24_1batch.om)
    --dataPath            (../../datasets/)
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/densenet24_1batch.om --dataPath=../../datasets/input_flair/,../datasets/input_t1/ --modelType=ID0121 --imgType=rgb
#    '
   exit 1
fi
loopNum=1
deviceId=0
imgType=rgb
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
rm -rf $cur_dir/../Benchmark/out/result.json

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%s")
mkdir -p $testcase_dir

#echo "======================preprocess======================"
#preprocess
echo "======================infer test======================"
infer_test
echo "======================collect results======================"
collect_result
echo "======================end======================"


