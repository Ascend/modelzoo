#!/bin/bash


cur_path=`pwd`/../
#source env.sh


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ALBERT-lcqmc-ZH_ID1461_for_TensorFlow"

RankSize=1
export RANK_SIZE=1

batch_size=32
#npu param
  
do_train=true   
do_eval=true    
max_seq_length=512
train_batch_size=32  
learning_rate=0.00001375 
max_predictions_per_seq=51
num_train_steps=12500
num_warmup_steps=12500
output_dir=./my_new_model_path  
save_checkpoints_steps=2000


# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path	 				data_path
    --ckpt_path					ckpt_path 
	--do_train  				do or not train 
	--do_eval   				do or not eval 
	--max_seq_length          	max_seq_length
	--train_batch_size 			train_batch_sizetr
	--learning_rate  			learning_rate
	--num_train_steps			num_train_steps
	--num_warmup_steps          num_warmup_steps
    --output_dir                output cur_path
    --save_checkpoints_steps    save_checkpoints_steps
    -h/--help		            show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --max_predictions_per_seq* ]];then
        max_predictions_per_seq=`echo ${para#*=}`
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
	elif [[ $para == --num_train_steps* ]];then
		num_train_steps=`echo ${para#*=}`
    elif [[ $para == --num_warmup_steps* ]];then
        num_warmup_steps=`echo ${para#*=}`
	elif [[ $para == --output_dir* ]];then
		output_dir=`echo ${para#*=}`
    elif [[ $para == --save_checkpoints_steps* ]];then
        save_checkpoints_steps=`echo ${para#*=}`
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

nohup python3 run_pretraining.py  \
        --input_file=${data_path}/tf*.tfrecord  \
        --do_train=$do_train  \
        --do_eval=$do_eval  \
        --train_batch_size=$train_batch_size  \
        --max_predictions_per_seq=$max_predictions_per_seq \
        --bert_config_file=$ckpt_path/albert_config_tiny.json \
        --max_seq_length=$max_seq_length  \
        --learning_rate=$learning_rate  \
        --num_train_steps=$num_train_steps \
        --num_warmup_steps=$num_warmup_steps \
        --save_checkpoints_steps=$save_checkpoints_steps \
        --output_dir=$output_dir \
        --init_checkpoint=$ckpt_path/albert_model.ckpt  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

# step_sec=`grep -a 'INFO:tensorflow:global_step/sec' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk 'END {print $2}'`
# Performance=`echo "scale=2; 1000/$step_sec" | bc`
# #echo "Final Precision MAP : $average_prec"
# echo "Final Performance ms/step : $Performance"
echo "Final Training Duration sec : $e2etime"  

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "examples/sec" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep -v "INFO"| awk 'END {print $6}'`
TrainingTime=`awk 'BEGIN {printf "%.2f\n", '1000'*'${batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "]   next_sentence_accuracy" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $7}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep -v "INFO"| awk -F  " " '{print $7}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep "INFO:tensorflow:loss = " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改' 
ActualLoss=(`awk 'END {print $NF}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log