set -x
export POD_NAME=another0

execpath=${PWD}

ulimit -c 0

start_time=`date +%s`
pip install opencv-python jupyter matplotlib tqdm
python -m pip install cityscapesscripts
export EXPERIMENTAL_DYNAMIC_PARTITION=1
#python train.py --update-mean-var --train-beta-gamma --random-scale --random-mirror --dataset cityscapes --filter-scale 1 --dataurl others
python3.7 train.py \
  --update-mean-var \
  --train-beta-gamma \
  --random-scale \
  --random-mirror \
  --dataset=cityscapes \
  --dataurl=others \
  --filter-scale=1 >train.log 2>&1
  #--model_dir=./model_path
end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="\[GEOP\]"  #功能检查字
key2="total loss"  #性能检查字
key3="val_loss" #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ] && [ `grep -c "$key3" "train.log"` -ne '0' ];then 
   echo "Get [GEOP] and total loss, Run testcase success!"   
else
   echo "Run testcase failed!"
fi
cat train.log
echo execution time was `expr $end_time - $start_time` s.