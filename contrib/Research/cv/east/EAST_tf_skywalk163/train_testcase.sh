set -x
export POD_NAME=another0

execpath=${PWD}
# rm -rf *.pbtxt
ulimit -c 0

python npu_train.py --training_data_path=$icdar2015_train > train.log 2>&1
# python npu_train.py  > train.log 2>&1

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="/[GEOP/]"  #功能检查字
key2="loss"
# change for /

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
cat train.log |grep $key2


