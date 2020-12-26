#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/../testlog/
dataset=$cur_dir/../datasets/
model=$cur_dir/../model/


if [  ! -d "$log_dir" ];then
    mkdir -p $log_dir
fi

if [ -f "$cur_dir/data.txt" ];then
    rm $cur_dir/data.txt
fi

preprocess()
{
    if [[ $modelType == yolov3 ]];then
        cd $cur_dir
        python3 image_demo.py
    fi

    if [[$modelType == 2D_lung]];then
        cd $cur_dir
        python3 2D_lung/lung_preprocess.py
    fi
}

infer_test()
{
    cd $cur_dir/../Benchmark/out/
    chmod +x benchmark
    if [[ $? -ne 0 ]];then
        echo "benchmark function execute failed."
        exit 1
    fi
        
    if [[ $useDvpp == 1 ]];then
        if [[ $imgType != raw ]];then
            echo "para imgType $imgType is not raw, the imgType must be raw if use dvpp"
            exit
        fi
    
    echo "./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --loopNum $loopNum --useDvpp $useDvpp"
    ./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --loopNum $loopNum --useDvpp $useDvpp
    
    elif [[ $imgType != yuv && $imgType != rgb ]];then
        echo "para imgType $imgType invalid, must be yuv or rgb with useDvpp==$useDvpp"
        exit 1
    else
        echo "./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --loopNum $loopNum --imgConfig $imgConfig --framework $framework"
        ./benchmark --dataDir $dataPath --om $modelPath --batchSize $batchSize --modelType $modelType --imgType $imgType --deviceId $deviceId --loopNum $loopNum --framework $framework --useDvpp $useDvpp
    fi
}

collect_result()
{
    
    cp $cur_dir/../Benchmark/out/test_perform_static.txt $testcase_dir/data.log
    
        
    AiModel_time=`cat $testcase_dir/data.log | tail -1 |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`
    AiModel_throughput=`cat $testcase_dir/data.log | tail -1 |awk '{print $8}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    
    InferenceEngine_total_time=`cat $testcase_dir/data.log |awk '{print $6}' |awk '{sum+=$1;cnt+=1}END{printf "%.2f\n", sum/cnt}'`    

    echo "InferencePerformance: $AiModel_time ms/batch, $AiModel_throughput images/sec" >> $testcase_dir/performance_results.log
    echo "InferenceTotalTime: $InferenceEngine_total_time ms" >> $testcase_dir/performance_results.log

    
    python $cur_dir/bintofloat.py $cur_dir/../results/${modelType}/  $outputType
    echo "start parse resnet" >> $testcase_dir/run.log
    python $cur_dir/result_statistical_tf.py $cur_dir/../results/${modelType}/ $trueValuePath >> $testcase_dir/run.log 2>&1 
    
    map=/
    top1=`cat $testcase_dir/run.log |grep "top1_accuracy_rate" | awk '{print $4}'`
    top5=`cat $testcase_dir/run.log |grep "top5_accuracy_rate" | awk '{print $6}'`
    echo top1: $top1 , top5: $top5  >> $testcase_dir/precision_results.log
   

    #write results to file
    #echo "$model $precision $cards $batchsize $AiModel_throughput $AiModel_time $top1 $top5 $map" >> $cur_dir/data.txt
    if [[ $imgType == yuv ]] || [[ $imgType == raw ]];then
        echo "${modelName} $AiModel_time $AiModel_throughput $map $top1 $top5" >> $cur_dir/data.txt
    elif [[ $imgType == rgb ]];then
        if [[ $inputType == uint8 ]];then
            echo "${modelName} $AiModel_time $AiModel_throughput $map $top1 $top5" >> $cur_dir/data.txt
        else
            echo "${modelName} $AiModel_time $AiModel_throughput $map $top1 $top5" >> $cur_dir/data.txt
        fi
    fi
    
    column -t $cur_dir/data.txt > $cur_dir/data1.txt
    mv $cur_dir/data1.txt $cur_dir/data.txt
    echo "model,throughput,performance_gpu,mPA@50%,top1,top5" > $cur_dir/data.csv
    cat $cur_dir/data.txt | tr -s '[:blank:]' ',' >> $cur_dir/data.csv
    
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
#    ./benchmark.sh --batchsize=1 --precision=fp16 --cardid=0 --omfile=/home/resnet50.om --data_dir=/home/imagenet/"
#
    echo ""
    echo 'parameter explain:
    --batchSize           Data number for one inference
    --modelType           model type(resnet50/resner18/resnet101/resnet152/yolov3)
    --imgType             input image format type(rgb/yuv/raw, default is yuv)
    --precision           precision type(fp16/fp32/int8, default is fp16)
    --inputType           input data type(fp32/fp16/int8, default is fp32)
    --outType             inference output type(fp32/fp16/int8, default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp, default is 0)
    --deviceId            running device ID(default is 0)
    --framework           (caffe，MindSpore,tensorflow, default is tensorflow)
    --modelPath           (../../model/resnet/resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32.om)
    --dataPath            (../../datasets/resnet/ImageNet2012_50000/)
    --trueValuePath       the path of true Value, default is ../../datasets/resnet/val_lable.txt
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --batchSize=1 --modelType=resnet50 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=../../model/resnet/resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32.om --dataPath=../../datasets/resnet/image-50000 --trueValuePath=../../datasets/resnet/val_lable.txt"
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
trueValuePath=../../datasets/inceptionv4/val_lable.txt

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
rm -rf $log_dir
rm -rf $cur_dir/data.*
rm -rf $cur_dir/../Benchmark/out/result*

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%S")
mkdir -p $testcase_dir 


echo "======================infer test========================"
infer_test
echo "======================collect results==================="
collect_result
echo "======================end==============================="
#echo "end time: $(date)" >> $testcase_dir/run.log


