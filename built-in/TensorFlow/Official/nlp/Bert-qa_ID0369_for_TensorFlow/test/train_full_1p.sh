#!bin/bash
cur_path=`pwd`

#集合通信
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

#数据集参数
data_path="."
more_path1="./uncased_L-12_H-768_A-12"
more_path2="./uncased_L-12_H-768_A-12"


#训练参数，需要根据模型修改
Network="Bert-qa_ID0369_for_TensorFlow"
num_train_steps=20
batch_size=32
#维测参数
overflow_dump=False
overflow_dump_path=$cur_path/output/overflow_dump
step_dump=False
step_dump_path=$cur_path/output/step_dump
check_loss_scale=Flase
#帮助提示，需要根据网络修改
if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=. --more_path1=./uncased_L-12_H-768_A-12 --more_path2=./uncased_L-12_H-768_A-12"
   exit 1
fi
#入参设置，需要根据网络修改
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
if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi
if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi
cd $cur_path
cd ..
rm -rf ./pretraining_output
start=$(date +%s)
nohup python3 run_pretraining.py \
  --input_file=$data_path/tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$more_path1/bert_config.json \
  --init_checkpoint=$more_path2/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
cp -r ./pretrain_output $cur_path/output/$ASCEND_DEVICE_ID

step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'$batch_size'*'1000'/'$step_sec'}'`
echo "Final Performance FPS : $FPS"
final_accuracy=`grep -a 'masked_lm_accuracy' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}'`

echo "Final accuracy : $final_accuracy"
echo "Final Training Duration sec : $e2etime"

################################性能看护#############################
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
grep masked_lm_loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${e2etime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${final_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
