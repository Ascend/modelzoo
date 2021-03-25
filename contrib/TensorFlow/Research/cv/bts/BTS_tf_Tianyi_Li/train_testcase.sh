set -x
export POD_NAME=another0

execpath=${PWD}
ulimit -c 0
export EXPERIMENTAL_DYNAMIC_PARTITION=1
python3.7 bts_main.py arguments_train_nyu.txt > train.log 2>&1
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字
key2="loss"


if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi
cat train.log