#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/
flipped_dataset=$cur_dir/../flipped_datasets/

if [ ! -d "$log_dir" ];then
	mkdir -p $log_dir
fi

if [ -f "$cur_dir/data.txt" ];then
	rm $cur_dir/data.txt
fi

preprocess()
{
	if [ ! -d "$dataset" ];then
		mkdir -p $dataset
	fi
	cd $cur_dir
	python3 data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset
	if [[ $flippedEval -eq 1 ]];then
	    if [ ! -d "$flipped_dataset" ];then
	        mkdir -p $flipped_dataset
	    fi
	    python3 data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset --flipped_eval --flipped_output_path=$flipped_dataset
	fi 
}

infer_test()
{
	cd $cur_dir/../Benchmark/out/
	chmod +x benchmark
	if [[ $? -ne 0 ]];then
		echo "benchmark function execute failed"
	exit 1
	fi
	echo "/benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp"
	./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp > performance.log
	if [[ $flippedEval -eq 1 ]];then
	    mv $cur_dir/../results $cur_dir/../results_noflip
	    echo "/benchmark --dataDir $flippedDataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp"
	    ./benchmark --dataDir $flippedDataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --framework $framework --useDvpp $useDvpp > performance_flipped.log
	fi
}

collect_result()
{
	cp $cur_dir/../Benchmark/out/test_perform_static.txt $testcase_dir/data.log
	AiModel_time=`cat $testcase_dir/data.log | tail -1 | awk '{print $6}' | awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
        AiModel_throughput=`cat $testcase_dir/data.log | tail -1 | awk '{print $8}' | awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
        InferenceEngine_total_time=`cat $testcase_dir/data.log | tail -1 | awk '{print $6}' | awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
	echo "InferencePerformance: $AiModel_time ms/batch, $AiModel_throughput images/sec" >> $testcase_dir/performance_results.log
	echo "InferenceTotalTime: $InferenceEngine_total_time ms" >> $testcase_dir/performance_results.log
	echo "start parse PSPnet" >> $testcase_dir/run.log

	if [[ $flippedEval -eq 1 ]];then
	    python3 $cur_dir/data_afterProcess.py --img_num=500 --result_path=$cur_dir/../results_noflip/${modelType}/ --data_dir=$cur_dir/../cityscapes --val_list=$cur_dir/../cityscapes/list/cityscapes_val_list.txt --flipped_eval --flipped_result_path=$cur_dir/../results/${modelType}/  >> $testcase_dir/run.log 2>&1
	else
	    python3 $cur_dir/data_afterProcess.py --img_num=500 --result_path=$cur_dir/../results/${modelType}/ --data_dir=$cur_dir/../cityscapes --val_list=$cur_dir/../cityscapes/list/cityscapes_val_list.txt >> $testcase_dir/run.log 2>&1
	fi

	mIoU=`cat $testcase_dir/run.log | grep "mIoU" | awk '{print $2}'`
	echo mIoU: $mIoU >> $testcase_dir/precision_results.log

	echo ""
	cat $testcase_dir/performance_results.log | grep InferencePerformance
	cat $testcase_dir/precision_results.log | tail -1
}

########################################################

if [[ $1 == --help || $1 == -h || $# -eq 0 ]];then
	echo "usage: ./benchmark_tf.sh <args>"
echo ""
echo 'parameter explain:
      --batchSize           Data number for one inference
      --modelType           model type(PSPnet101)
      --imgType             input image format type(rgb/yuv/raw/,default is yuv)
      --precision           precision type(fp16/fp32/int8,default is fp16)
      --inputType           input data type(fp32/fp16/int8,default is fp32) 
      --outType             inference output type(fp32/fp16/int8,default is fp32)
      --useDvpp             (0-no dvpp,1--use dvpp,default is 0)
      --deviceId            running device ID(default is 0)
      --framework            (caffe,MindSpore,tensorflow,default is tensorflow
      --modelPath           (../../model/pspnet101_1batch.om)
      --dataPath            (../../datasets/)
      --flippedEval         flipped img(default is 0)
      --flippedDataPath     (../../flipped_datasets/)
      -h/--help             Show help message
      Example:
      ./benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb
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
flippedEval=0

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
	elif [[ $para == --flippedEval* ]];then
		flippedEval=`echo ${para#*=}`
	elif [[ $para == --flippedDataPath* ]];then
		flippedDataPath=`echo ${para#*=}`
	fi
done
rm -rf $cur_dir/../results
rm -rf $cur_dir/../results_noflip
rm -rf $cur_dir/../Benchmark/out/result*
rm -rf $cur_dir/../Benchmark/out/performance.log
rm -rf $cur_dir/../Benchmark/out/performance_flipped.log
rm -rf $log_dir
testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%s")
mkdir -p $testcase_dir
echo "======================preprocess test======================"
preprocess
echo "======================infer test======================"
infer_test
echo "======================collect test======================"
collect_result
echo "======================end======================"



