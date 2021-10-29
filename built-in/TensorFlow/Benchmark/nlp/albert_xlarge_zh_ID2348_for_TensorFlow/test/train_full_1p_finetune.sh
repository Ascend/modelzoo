#!/bin/bash

cur_path=`pwd`/../

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="albert_xlarge_zh_ID2355_for_TensorFlow"
export JOB_ID=10087
RANK_SIZE=1

#npu param
task_name=lcqmc_pair   
do_train=true   
do_eval=true    
max_seq_length=128
train_batch_size=32   
learning_rate=1e-6    
num_train_epochs=3
output_dir=./albert_lcqmc_checkpoints 

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path	 				data_path
    --ckpt_path					ckpt_path
    --task_name  			    task_name
	--do_train  				do or not train 
	--do_eval   				do or not eval 
	--max_seq_length          	max_seq_length
	--train_batch_size 			train_batch_sizetr
	--learning_rate  			learning_rate
	--num_train_epochs			num_train_epochs
	--output_dir                output cur_path
    -h/--help		            show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --task_name* ]];then
        task_name=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    elif [[ $para == --do_train* ]];then
		do_train=`echo ${para#*=}`
	elif [[ $para == --do_eval* ]];then
		do_eval=`echo ${para#*=}`
	elif [[ $para == --max_seq_length* ]];then
		max_seq_length=`echo ${para#*=}`
	elif [[ $para == --train_batch_size* ]];then
		train_batch_size=`echo ${para#*=}`
	elif [[ $para == --learning_rate* ]];then
		learning_rate=`echo ${para#*=}`
	elif [[ $para == --num_train_epochs* ]];then
		num_train_epochs=`echo ${para#*=}`
	elif [[ $para == --output_dir* ]];then
		output_dir=`echo ${para#*=}`
    fi
done

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
cd $cur_path
start=$(date +%s)

nohup python3 -u run_classifier.py   \
         --task_name=$task_name   \
         --do_train=$do_train   \
         --do_eval=$do_eval   \
         --data_dir=$data_path   \
         --vocab_file=./albert_config/vocab.txt  \
         --bert_config_file=./albert_config/albert_config_xlarge.json \
         --max_seq_length=$max_seq_length \
         --train_batch_size=$train_batch_size   \
         --learning_rate=$learning_rate  \
         --num_train_epochs=$num_train_epochs \
         --output_dir=$output_dir \
         --init_checkpoint=$ckpt_path/albert_model.ckpt  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
echo "Final Training Duration sec : $e2etime"  

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "examples/sec" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep -v "INFO"| awk 'END {print $6}'`
TrainingTime=`awk 'BEGIN {printf "%.2f\n", '${train_batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "eval_accuracy" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

##下面字段用于冒烟看护
BatchSize=${train_batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "INFO:tensorflow:global_step " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F  " " '{print $6}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改' 
ActualLoss=(`awk 'END {print $NF}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
