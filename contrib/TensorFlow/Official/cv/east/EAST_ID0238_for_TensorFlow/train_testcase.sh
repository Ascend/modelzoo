#How to download pretrain_model and dataset?
#Need resnet_v1_50.ckpt model and icdar2015

python3.7 npu_train.py \
--max_steps=20  \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=$icdar2015_train \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=$resnetv1_50_ckpt >train.log 2>&1

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="[GEOP]"  #功能检查字
key2="total loss"  #性能检查字
key3="xxx"  #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi