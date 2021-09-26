#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`
#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087

RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ShuffleNetV1-1.0x-group3_ID2129_for_TensorFlow"
batch_size=64
epochs=100
RankSize=1
RANK_ID_START=0
#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
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

#训练开始时间，不需要修改
start_time=$(date +%s)
cd $cur_path/../
mkdir -p logs/checkpoints/

train_path=${data_path}/generated_tfrecord/
val_path=${data_path}/generated_tfrecord_validation/
#参数修改
sed -i "s|NUM_EPOCHS = 100|NUM_EPOCHS = ${epochs}|g" train_shufflenet.py
sed -i "s|'train_dataset_path': 'data/jiu-huan_train/'|'train_dataset_path': '${train_path}'|g" train_shufflenet.py
sed -i "s|'val_dataset_path': 'data/jiu-huan_val/'|'val_dataset_path': '${val_path}'|g" train_shufflenet.py
wait

#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

#    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
#    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
#    nohup python3.7 $cur_path/../train_adda_seg.py ${data_path}/data/inria_test source_image source_label_index target_image adda_deeplab_v3p.h5 \
#        --optimizer adam \
#		--base_learning_rate 1e-4 \
#		--min_learning_rate 1e-7 \
#		--image_width 256 \
#		--image_height 256 \
#		--image_channel 3 \
#		--image_suffix .png \
#		--label_suffix .png \
#		--n_class 2 \
#		--batch_size 2 \
#		--iterations 50 \
#		--weight_decay 1e-4 \
#		--initializer he_normal \
#		--bn_epsilon 1e-3 \
#		--bn_momentum 0.99 \
#		--pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 \
#		--source_fname_file ${data_path}/data/inria_test/source.txt \
#		--target_fname_file ${data_path}/data/inria_test/target.txt \
#		--logs_dir ./logs \
#		--augmentations flip_x,flip_y,random_crop \
#		--display 1 \
#		--snapshot 5   > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

    echo "data_path is : "
    echo "data_path is : $data_path"
	#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 $cur_path/../train_shufflenet.py  --data_path ${data_path} --epoch ${epochs}> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#参数回改
sed -i "s|NUM_EPOCHS = ${epochs}|NUM_EPOCHS = 100|g" train_shufflenet.py
sed -i "s|'train_dataset_path': '${train_path}'|'train_dataset_path': 'data/jiu-huan_train/'|g" train_shufflenet.py
sed -i "s|'val_dataset_path': '${val_path}'|'val_dataset_path': 'data/jiu-huan_val/'|g" train_shufflenet.py
wait

#sed -i "s|${data_path}/tmp|tmp|g" train.py
cd test

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改 
TrainingTime=`grep -nrs "loss =" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v top|awk -F 'step' '{print $2}'|tr -d '('|awk 'END {print $3}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'50'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

##输出训练精度,需要模型审视修改
##train_accuracy=`grep 'accuracy =' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $9}'|awk -F "," '{print $1}'`
#train_accuracy=`grep 'accuracy =' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $9}'|awk -F , '{print $1}'`
train_accuracy=`grep 'accuracy' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $9}'|awk -F "," END'{print $1}'`
##打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'acc'

#获取性能数据
##吞吐量，不需要修改
ActualFPS=${FPS}
##单迭代训练时长，不需要修改
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -nrs "loss =" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep -v top|tr -d ','|awk '{print $3}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log