python multigpu_train.py \
		--checkpoint_path='./checkpoint' \
		--text_scale=512 \
		--training_data_path=$icdar2015_train \
		--geometry=RBOX \
		--learning_rate=0.0001 \
		--num_readers=24 \
		--max_steps=20 \
		--save_checkpoint_steps=10 >train.log 2>&1
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字
key1="total loss"  #功能检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

