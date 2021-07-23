#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/

if [  ! -d "$log_dir" ];then
    mkdir -p $log_dir
fi

preprocess()
{
	if [ ! -d "$dataset" ];then
		mkdir -p $dataset
	fi
	cd $cur_dir
	python3 ssd_dataPrepare.py --input_file_path=../coco_minival2014 --output_file_path=$dataset --crop_width=640 --crop_height=640 --save_conf_path=./img_info
	
}
infer_test()
{
	cd $cur_dir/../Benchmark/out/
	chmod +x benchmark
	if [[ $? -ne 0 ]];then
        echo "benchmark function execute failed."
        exit 1
    fi
	
	echo "./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp"
	./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp > performance.log
}

collect_result()
{
	cp $cur_dir/../Benchmark/out/test_perform_static.txt $testcase_dir/data.log
	
	AiModel_time=`cat $testcase_dir/data.log | tail -1 |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    AiModel_throughput=`cat $testcase_dir/data.log | tail -1 |awk '{print $8}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    
    InferenceEngine_total_time=`cat $testcase_dir/data.log |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    

    echo "InferencePerformance: $AiModel_time ms/batch, $AiModel_throughput images/sec" >> $testcase_dir/performance_results.log
    echo "InferenceTotalTime: $InferenceEngine_total_time ms" >> $testcase_dir/performance_results.log
	
	python3 $cur_dir/coco_afterProcess.py --result_file_path=$cur_dir/../results/${modelType}/ --img_conf_path=$cur_dir/img_info > file.log
	
	echo "start parse ssd_resnet50fpn" >> $testcase_dir/run.log
	
	python3 $cur_dir/eval_coco.py $cur_dir/../Benchmark/out/result.json $trueValuePath >> $testcase_dir/run.log 2>&1
	
	mAp=`cat $testcase_dir/run.log |grep 'Average Precision' |grep 'IoU=0.50:0.95' |grep 'area=   all' | awk '{print $13}'`
	
	echo mAp: $mAp  >> $testcase_dir/precision_results.log
	
	#print results
    echo ""
    cat $testcase_dir/performance_results.log |grep InferencePerformance 
    cat $testcase_dir/precision_results.log |tail -1
}
###########################START#########################

if [[ $1 == --help || $1 == -h || $# -eq 0 ]];then 
    echo "usage: ./benchmark.sh <args>"
#    echo "example:
#    Execute the following command when model file is converted:
#    ./benchmark.sh --modelPath=/home/mobilenetv3_large.om --data_dir=/home/imagenet/ --batchsize=1 --outputType=fp32"
#

echo ""
    echo 'parameter explain:
    --batchSize           Data number for one inference
	--modelType           model type(ssd_resnet50_fpn)
    --imgType             input image format type(rgb/yuv/raw, default is yuv)
    --precision           precision type(fp16/fp32/int8, default is fp16)
    --inputType           input data type(fp32/fp16/int8, default is fp32)
    --outType             inference output type(fp32/fp16/int8, default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp, default is 0)
    --deviceId            running device ID(default is 0)
    --framework           (caffe，MindSpore,tensorflow, default is tensorflow)
    --modelPath           (../../model/ssd_resnet50_fpn.om)
    --dataPath            (../../datasets/)
    --trueValuePath       the path of true Value, default is ../../datasets/input_5w.csv
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/ssd_resnet50_fpn.om --dataPath=../../datasets/ --modelType=ssd_resnet50_fpn --imgType=rgb --trueValuePath=../../scripts/instances_minival2014.json" 
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
trueValuePath=scripts/instances_minival2014.json

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

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%S")
mkdir -p $testcase_dir

echo "======================infer test========================"
infer_test
echo "======================collect results==================="
collect_result
echo "======================end==============================="
#echo "end time: $(date)" >> $testcase_dir/run.log
