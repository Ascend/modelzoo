#!/bin/bash
# ǰ·Ҫ޸
cur_path=`pwd`/../

#Ĭ־,Ҫ޸
#export ASCEND_GLOBAL_LOG_LEVEL=3

#Ҫģ޸
#Batch Size
batch_size=16
#ƣͬĿ¼
Network="SRGAN_ID1881_for_TensorFlow"
#DeviceĬΪ1
RANK_SIZE=1
#ѵepochѡ
train_epochs=2000
#ѵstep
train_steps=50000
#ѧϰ
learning_rate=1e-5

#
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_1p.sh"
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

##############ִѵ##########
cd $cur_path

##############׼ļ##########
cp -r /npu/traindata/ID1881_CarPeting_TF_SRGAN/checkpoint/ ${cur_path}
cp /npu/traindata/ID1881_CarPeting_TF_SRGAN/vgg19.npy ${cur_path}

#޸
n_iteration=55
sed -i "s|config.TRAIN.hr_img_path = '|config.TRAIN.hr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.lr_img_path = '|config.TRAIN.lr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.VALID.hr_img_path = '|config.VALID.hr_img_path = '${data_path}/|g" ${cur_path}/config.py
sed -i "s|config.VALID.lr_img_path = '|config.VALID.lr_img_path = '${data_path}/|g" ${cur_path}/config.py

wait

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 main.py > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

# ɾļ
if [ -d $cur_path/checkpoint/ ];then
	rm -rf ${cur_path}/checkpoint/
fi

if [ -e $cur_path/vgg19.npy ];then
	rm -rf ${cur_path}/vgg19.npy 
fi

#ظ
sed -i "s|config.TRAIN.hr_img_path = '${data_path}/|config.TRAIN.hr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.TRAIN.lr_img_path = '${data_path}/|config.TRAIN.lr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.VALID.hr_img_path = '${data_path}/|config.VALID.hr_img_path = '|g" ${cur_path}/config.py
sed -i "s|config.VALID.lr_img_path = '${data_path}/|config.VALID.lr_img_path = '|g" ${cur_path}/config.py

#ӡҪ޸
echo "------------------ Final result ------------------"
#FPSҪģ޸
TrainingTime=`grep "final_time" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' | sed 's|s,||g'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'${n_iteration}'*'${batch_size}'/'${TrainingTime}'}'`
#ӡҪ޸
echo "Final Performance images/sec : $FPS"

#ѵ,Ҫģ޸
#train_accuracy=`grep "train_acc " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $12}'|sed 's/,//g'`
#ӡҪ޸
#echo "Final Train Accuracy : ${train_accuracy}"


#ܿ
#ѵϢҪ޸
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##ȡݣҪ޸
#
ActualFPS=${FPS}
#ѵʱ
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#train_$ASCEND_DEVICE_ID.logȡLosstrain_${CaseName}_loss.txtУҪģ
grep "mse" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ',' '{print $2}'|awk '{print $2}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#һlossֵҪ޸
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#ؼϢӡ${CaseName}.logУҪ޸
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

# ˳anaconda
if [ -n "$conda_name"];then
   conda deactivate
fi
