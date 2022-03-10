#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改
export RANK_SIZE=8
export RANK_TABLE_FILE=$cur_path/test/${RANK_SIZE}p.json
export JOB_ID=10087
export DEVICE_INDEX=0
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP_ETP=3


#基础参数，需要模型审视修改
#Batch Size
batch_size=16
#网络名称，同目录名称
Network="SRGAN_ID1881_for_TensorFlow"
#Device数量，单卡默认为1
#训练epoch，可选
train_epochs=10
#训练step
train_steps=50000
#学习率
learning_rate=1e-5

#参数配置
data_path="/npu/traindata/ID1881_CarPeting_TF_SRGAN"


#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
   
   if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
	  source set_conda.sh
	  source activate $conda_name
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi


##############执行训练##########
cd $cur_path

##############准备所需文件##########
cp -r /npu/traindata/ID1881_CarPeting_TF_SRGAN/checkpoint/ ${cur_path}
cp /npu/traindata/ID1881_CarPeting_TF_SRGAN/vgg19.npy ${cur_path}

#参数修改
n_iteration=55
sed -i "s|config.TRAIN.n_epoch_init = 100|config.TRAIN.n_epoch_init = 10|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.n_epoch = 2000|config.TRAIN.n_epoch = 10|g" ${cur_path}/config.py

sed -i "s|config.TRAIN.hr_img_path = '|config.TRAIN.hr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.lr_img_path = '|config.TRAIN.lr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.VALID.hr_img_path = '|config.VALID.hr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.VALID.lr_img_path = '|config.VALID.lr_img_path = '${data_path}/|g" ${cur_path}/config.py

wait


#############执行训练#########################
start=$(date +%s)
cd $cur_path/../
for i in 0 1 2 3 4 5 6 7
do
    export RANK_ID=$i
    export ASCEND_DEVICE_ID=$i
    echo "Device ID : $ASCEND_DEVICE_ID"
    if [ -d $cur_path/test/output/$ASCEND_DEVICE_ID ];then
       rm -rf $cur_pat/test/output/$ASCEND_DEVICE_ID
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    else
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    fi
    cd $cur_path
    #nohup ${bind_core} python3 demo.py --weights_path=${ckpt_path} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
        nohup python3 main.py --npu_nums=8> $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end=$(date +%s)
e2et_ime=$(( $end - $start ))



#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

# 删除数据文件
if [ -d $cur_path/checkpoint/ ];then
	rm -rf ${cur_path}/checkpoint/
fi

if [ -e $cur_path/vgg19.npy ];then
	rm -rf ${cur_path}/vgg19.npy 
fi

#参数回改
sed -i "s|config.TRAIN.n_epoch_init = 10|config.TRAIN.n_epoch_init = 100|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.n_epoch = 10|config.TRAIN.n_epoch = 2000|g" ${cur_path}/config.py

sed -i "s|config.TRAIN.hr_img_path = '${data_path}/|config.TRAIN.hr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.lr_img_path = '${data_path}/|config.TRAIN.lr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.VALID.hr_img_path = '${data_path}/|config.VALID.hr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.VALID.lr_img_path = '${data_path}/|config.VALID.lr_img_path = '|g" ${cur_path}/config.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "final_time" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' | sed 's|s,||g'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'${n_iteration}'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "train_acc " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $12}'|sed 's/,//g'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "mse" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ',' '{print $2}'|awk '{print $2}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = None" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

# 退出anaconda环境
if [ -n "$conda_name"];then
   conda deactivate
fi





















