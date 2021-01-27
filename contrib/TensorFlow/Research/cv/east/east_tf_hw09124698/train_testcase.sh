set -x
export POD_NAME=another0

execpath=${PWD}
pip3 install scipy
output="./output"
batch_size=64
max_steps=20
rm -rf *.pbtxt
ulimit -c 0

start_time=`date +%s`
python multigpu_train.py \
		--gpu_list=0 \
		--input_size=512 \
		--batch_size_per_gpu=$batch_size \
		--checkpoint_path=$output \
		--text_scale=512 \
		--training_data_path=$icdar2015_train \
		--geometry=RBOX \
		--learning_rate=0.0001 \
		--num_readers=24 \
		--max_steps=$max_steps \
		--save_checkpoint_steps=10 >train.log 2>&1

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字


if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "Step" "train.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi
if  ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi
cat train.log |grep Step

echo execution time was `expr $end_time - $start_time` s.