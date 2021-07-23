#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/testlog/
dataset=$cur_dir/datasets/
dataset_bin=$cur_dir/datasets_bin/

if [  ! -d "$log_dir" ];then
    mkdir -p $log_dir
fi

preprocess()
{
	if [ ! -d "$dataset" ];then
		mkdir -p $dataset
	fi
        if [ ! -d "$dataset_bin" ];then
                mkdir -p $dataset_bin
        fi
	cd $cur_dir/script
	python3 align/align_dataset_mtcnn_facereset.py $cur_dir/lfw $dataset
        python3 preprocess.py $cur_dir/config/basemodel.py $dataset_bin
	
}
infer_test()
{
	cd $cur_dir/Benchmark/out/
	chmod +x benchmark
	if [[ $? -ne 0 ]];then
        echo "benchmark function execute failed."
        exit 1
        fi
	#/usr/local/Ascend/atc/bin/atc --model ./model/face_resnet50_tf.pb   --framework=3  --output=face_resnet50 --input_shape="image_batch:1,112,96,3" --enable_small_channel=1 --soc_version=Ascend310
	echo "$cur_dir/Benchmark/out/benchmark --dataDir $dataset_bin --om $modelPath  --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp > $testcase_dir/performance.log"
              $cur_dir/Benchmark/out/benchmark --dataDir $dataset_bin --om $modelPath  --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp > $testcase_dir/performance.log
}

collect_result()
{
	cp $cur_dir/Benchmark/out/test_perform_static.txt $testcase_dir/data.log
	
	AiModel_time=`cat $testcase_dir/data.log | tail -1 |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    AiModel_throughput=`cat $testcase_dir/data.log | tail -1 |awk '{print $8}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    
    InferenceEngine_total_time=`cat $testcase_dir/data.log |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    

    echo "InferencePerformance: $AiModel_time ms/batch, $AiModel_throughput images/sec" >> $testcase_dir/performance_results.log
    echo "InferenceTotalTime: $InferenceEngine_total_time ms" >> $testcase_dir/performance_results.log
	cd $cur_dir/script
	python3 afterprocess.py $cur_dir/config/basemodel.py $cur_dir/results/${modelType}/ > $testcase_dir/file.log
	
	echo "start parse faceresnet50" >> $testcase_dir/run.log
	
	
	Accu=`cat $testcase_dir/file.log |grep 'Embeddings Accuracy' | awk '{print $3}'`
	
	echo Embeddings Accuracy: $Accu  >> $testcase_dir/precision_results.log
	
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
    -h/--help             Show help message
    Example:
     ./benchmark_tf.sh --batchSize=1 --modelPath=/home/liukongyuan/ID1372_face_resnet50/pure/model/face_resnet50.om --dataPath=$dataset_bin --modelType=faceresnet50 --imgType=rgb"
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
    fi
done

rm -rf $cur_dir/results
rm -rf $cur_dir/file.log
rm -rf $log_dir
rm -rf $cur_dir/Benchmark/out/result*
rm -rf $cur_dir/Benchmark/out/modelInputOutputInfo
rm -rf $cur_dir/Benchmark/out/test_perform_static.txt
rm -rf $cur_dir/Benchmark/out/performance.log
rm -rf $cur_dir/Benchmark/out/result.json
rm -rf $cur_dir/test_perform_static.txt
rm -rf $cur_dir/modelInputOutputInfo

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%S")
mkdir -p $testcase_dir

echo "======================Data preprocessing========================"
preprocess
echo "======================infer test========================"
infer_test
echo "======================collect results==================="
collect_result
echo "======================end==============================="
#echo "end time: $(date)" >> $testcase_dir/run.log
