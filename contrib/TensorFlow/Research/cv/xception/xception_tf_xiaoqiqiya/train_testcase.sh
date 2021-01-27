start_time=`date +%s`
python3  run_xception.py  \
--model_path $XCEPTION_CKPT/xception_model.ckpt \
--data_path $IMG2012_val_299  \
--output_path  ./model_save  \
--do_train True  \
--image_num  1280 \
--class_num  1000  \
--batch_size  64 \
--epoch 1 \
--learning_rate  0.01   \
--save_checkpoints_steps  10 >train.log 2>&1
end_time=`date +%s`

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="train loss ="  #功能检查字


if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.
          
