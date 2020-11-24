#! /bin/bash
#Ascend社区已预置的数据集、预训练模型、ATC-OM模型等
DATA_PATH=$WMT_ENDE
VOCAB_SOURCE=${DATA_PATH}/vocab.share
VOCAB_TARGET=${DATA_PATH}/vocab.share
TRAIN_FILES=${DATA_PATH}/concat128/train.l128.tfrecord-001-of-016

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
#obsutil cp obs://obsxxx/xxx/xxx.ckpt ./model/ -f -r

#testcase主体，开发者根据不同模型写作
bash train_1000step.sh > train.log
#测试用例只执行1000个step，并保存打印信息至train.log

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="xxx"  #功能检查字
key2="xxx"  #性能检查字
key3="xxx"  #精度检查字


if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi