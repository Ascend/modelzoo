set -x
export POD_NAME=another0

execpath=${PWD}

output="./output_base_v2"
rm -rf *.pbtxt
ulimit -c 0

start_time=`date +%s`
python3.7 train.py \
    --checkpoint_path=./checkpoint/ \
	--text_scale=512 \
	--training_data_path=$icdar2015_train \
	--geometry=RBOX \
	--learning_rate=0.0001 \
	--num_readers=24 \
	--max_steps=20 \
    --pretrained_model_path=$resnetv1_50_ckpt \
	--save_checkpoint_steps=10 >train.log 2>&1
    #--model_dir=./model_path

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字
key2="loss"  #性能检查字
#key3="val_loss"  #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

if [ `grep -c "$key2" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

cat train.log |grep loss

echo execution time was `expr $end_time - $start_time` s.
