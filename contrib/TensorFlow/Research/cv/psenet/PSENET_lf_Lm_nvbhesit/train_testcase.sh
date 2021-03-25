set -x
export POD_NAME=another0
#export datapath=/data/dataset/storage/icdar/
execpath=${PWD}
output="./resnet_train/"
#cd $output
#rm *
#cd $execpath
#echo $execpath

sudo -H python -m pip --default-timeout=100 install pyclipper

#rm -rf *.pbtxt
ulimit -c 0
start_time=`date +%s`
python3.7 train.py \
	--training_data_path=$icdar2015_train \
	--checkpoint_path=$output \
	--num_readers=24 \
	--input_size=512 \
	--max_steps=1000 \
	--learning_rate=0.0001 \
	--save_checkpoint_steps=100 \
	--save_summary_steps=10 >train.log 2>&1
	#--model_dir=./model_path
end_time=`date +%s`
key1="\[GEOP\]"
key2="total loss"
#key3="val_loss"
if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ] ;then
	echo "Run testcase success!"
else
	echo "Run testcase failed!"
fi
cat train.log | grep "total loss"
echo execution time was `expr $end_time - $start_time` s.
