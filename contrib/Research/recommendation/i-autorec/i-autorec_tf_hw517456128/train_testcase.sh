set -x
export POD_NAME=another0

execpath=${PWD}

output='./output'
rm -rf *.pbtxt
ulimit -c 0

start_time=`date +%s`
python3.7 -m main.py \
	--data_url $ml1m_dataset \
	--train_url $output \
	--hidden_neuron 1024 \
	--train_epoch 10 \
	--batch_size 256

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字
key2="xxx"  #性能检查字
key3="xxx"  #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.
