#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`
#集合通信参数,不需要修改


export RANK_SIZE=8


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ResnetVariant_ID0033_for_TensorFlow"
batch_size=256


#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done


#data_path='../'
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


#修改参数
##sed -i "s|max_steps=len(self\.train\_loader) \* self\.epochs|max_steps=$train_steps|g"  $cur_path/../automl/vega/core/trainer/trainer.py
#sed -i "s|./automl|$cur_path/../automl|g"  $cur_path/../automl/examples/run_example.py
#sed -i "53s|epochs: 2|epochs: $train_epochs|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
#sed -i "94s|epochs: 100|epochs: $train_epochs|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
#sed -i "s|/root/datasets/imagenet_tfrecord|$data_path|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
sed -i "s|/root/projects/1.0/automl/examples/tasks|$cur_path|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf_8p.yml
sed -i "s|/root/datasets/imagenet_tfrecord|$data_path|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf_8p.yml

sed -i 's|cfg_file = "./automl/examples/nas/backbone_nas/backbone_nas_tf_8p.yml"|cfg_file = sys.argv[1]|g'  $cur_path/../automl/examples/run_example.py

wait
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../automl/examples/
ASCEND_DEVICE_ID=0
    
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi
bash run_resnetvariant_8p.sh |& tee ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#参数改回
#sed -i "s|max_steps=$train_steps|max_steps=len(self\.train\_loader) \* self\.epochs|g"  $cur_path/../automl/vega/core/trainer/trainer.py
#sed -i "s|$cur_path/../automl|./automl|g"  $cur_path/../automl/examples/run_example.py
#sed -i "53s|epochs: $train_epochs|epochs: 2|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
#sed -i "94s|epochs: $train_epochs|epochs: 100|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
#sed -i "s|$data_path|/root/datasets/imagenet_tfrecord|g"  $cur_path/../automl/examples/nas/backbone_nas/backbone_nas_tf.yml
#wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep 'INFO loss' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'sec' '{print $1}'|tr -d '('|awk 'NR>2'|awk '{print $NF}'|awk '{sum+=$1} END {print  1000*sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000*10/'${step_time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

train_accuracy=`grep -r accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep metrics|awk '{print $8}'|awk -F "}" '{sum+=$1} END {print "AvgAcc =", sum/NR}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$step_time


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'INFO loss' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'loss' '{print $2}'|tr -d ','|awk '{print $2}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/0/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log