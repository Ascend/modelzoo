#!bin/bash
#当前路径,不需要修改
cur_path=`pwd`
#集合通信
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

#数据集参数
data_path="."


#训练参数，需要根据模型修改
Network="GraphSAGE_ID0089_for_TensorFlow"
# num_train_steps=20
batch_size=512
# 维测参数
overflow_dump=False
overflow_dump_path=$cur_path/output/overflow_dump
step_dump=False
step_dump_path=$cur_path/output/step_dump
check_loss_scale=Flase
#帮助提示，需要根据网络修改
if [[ $1 == --help || $1 == --h ]];then
    echo "usage:./train_full_1p.sh --data_path=/npu/traindata/ppi_for_GraphSAGE"
    exit 1
fi
#入参设置，需要根据网络修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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
echo "Device ID: $ASCEND_DEVICE_ID"
start_time=$(date +%s)
echo "$data_path/ppi"
nohup python3 $cur_path/../graphsage/supervised_train.py \
    --train_prefix $data_path/ppi/ppi \
	--model meanpool \
	--base_log_dir $cur_path/output/$ASCEND_DEVICE_ID/ppi \
	--device npu \
	--device_ids $ASCEND_DEVICE_ID \
	--rank_size 1 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
cp $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log /home/hzh
sec_step=`grep Runtime $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep supervised|awk '{print $6}'`
FPS=`python3 $cur_path/eval_FPS.py $sec_step $batch_size`
echo "Final Performance FPS : $FPS"
final_accuracy=`tail -n1 $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk '{print $8}'|awk -F = '{print $2}'|awk -F , '{print $1}'`

echo "Final accuracy : $final_accuracy"
echo "Final Training Duration sec : $e2e_time"

################################精度看护#############################
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
ActualLoss=`grep loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tail -1|awk '{print $7}'|awk -F = '{print $2}'|awk -F , '{print $1}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
grep loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tail -1|awk '{print $7}'|awk -F = '{print $2}'|awk -F , '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${final_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
