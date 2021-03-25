start_time=`date +%s`
python3  train.py  \
--HR_data 's3://esrgan-ascend/npz/DIV2K/DIV2K/HR_sub' \
--LR_data 's3://esrgan-ascend/npz/DIV2K/DIV2K/LR_sub'   \
--VGG19_weights 's3://esrgan-ascend/pre_trained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5' \
--pre_train_checkpoint_dir  'pre_train_checkpoint' \
--num_iter 1000000 >train.log 2>&1
end_time=`date +%s`

echo execution time was `expr $end_time - $start_time` s.

