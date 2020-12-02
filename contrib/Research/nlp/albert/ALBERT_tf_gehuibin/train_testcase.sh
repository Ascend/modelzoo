set -x
export POD_NAME=another0

execpath=${PWD}

output="./output_base_v2"
rm -rf *.pbtxt
ulimit -c 0

start_time=`date +%s`
python3.7 -m run_squad_v2 \
  --output_dir=$output \
  --input_dir=$SQUADV2 \
  --model_dir=$ALBERT_CKPT/albert_base_v2 \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train \
  --train_batch_size=32 \
  --predict_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=500 \
  --n_best_size=20 \
  --max_answer_length=30 >train.log 2>&1
  #--model_dir=./model_path

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="[GEOP]"  #功能检查字
key2="xxx"  #性能检查字
key3="xxx"  #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.