#!/bin/bash
cur_path=`pwd`/../
#export ASCEND_DEVICE_ID=0

data_path="."
more_path1="./uncased_L-12_H-768_A-12"
more_path2="./uncased_L-12_H-768_A-12"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=. --more_path1=./uncased_L-12_H-768_A-12 --more_path2=./uncased_L-12_H-768_A-12"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   elif [[ $para == --more_path1* ]];then
      more_path1=`echo ${para#*=}`
   elif [[ $para == --more_path2* ]];then
      more_path2=`echo ${para#*=}` 
   fi
done
   
cd $cur_path
rm -rf ./pretraining_output

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait
batch_size=32
start=$(date +%s)
nohup python3 run_pretraining.py \
  --input_file=$data_path/tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$more_path1/bert_config.json \
  --init_checkpoint=$more_path2/bert_model.ckpt \
  --train_batch_size=$batch_size \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
cp -r $cur_path/pretraining_output $cur_path/test/output/$ASCEND_DEVICE_ID

step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
average_perf=`awk 'BEGIN{printf "%.2f\n",'$batch_size'*'$step_sec'}'`

echo "Final Performance FPS : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
##定义网络基本信息
#网络名称，同目录名称
Network="Bert-qa_ID0369_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#BatchSize
BatchSize=32
#设备类型，自动获取，此处无需修改
DeviceType=`uname -m`
#用例名称，自动获取，此处无需修改
CaseName=${Network}_${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据
#吞吐量
ActualFPS=$average_perf
#单迭代训练时长
TraingingTime=$e2etime

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到${CaseName}_loss.txt中，需要修改***匹配规则
grep "INFO:tensorflow:Saving dict for global step 20: global_step = 20, loss =" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F ',' '{print $2}' | awk -F '=' '{print $2}'> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.log
#最后一个迭代Loss值
ActualLoss=`grep "INFO:tensorflow:Saving dict for global step 20: global_step = 20, loss =" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F ',' '{print $2}' | awk -F '=' '{print $2}'`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}"  >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TraingingTime = ${TraingingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log