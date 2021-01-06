start_time=`date +%s`
python3  run_resnet.py  \
--train_Sharp_path "s3://deblurgan/data/tain/sharp" \
--train_Blur_path "s3://deblurgan/data/tain/blur"   \
--vgg_path  "s3://deblurgan/pre_train_model/vgg19.npy" \
--model_path "./model"  \
--max_epoch  300 >train.log 2>&1
end_time=`date +%s`

echo execution time was `expr $end_time - $start_time` s.
