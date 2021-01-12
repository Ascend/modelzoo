#! /bin/bash
# the path of the original validation images
source_img_path="/home/HwHiAiUser/ilsvrc2012/val"

# model="/home/HwHiAiUser/mobilenetv3/om_model_official/device/mobilenet_v3_large.om"
model="/home/HwHiAiUser/mobilenetv3/om_model/device/mobilenet_v3_large.om"

input="input" # the preprocessed image bin files will be saved here.
output="output" # the inference results
label="ground_truth/val_map.txt" # gt 

# preprocess
# python3 img_preprocess.py --src_path=$source_img_path --dst_path=$input 
#start inference
# ./msame --model $model --input $input --output $output 2>&1 | tee logs/inference.log
#top1 accuarcy
python3 accuarcy_top1.py $output $label 2>&1 | tee logs/top1.log
#
# 结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
avg_time=`grep "Inference average time without first time:" logs/inference.log | awk '{print $7}'`
top1=`grep "Top1 accuarcy:" logs/top1.log | awk '{print $7}'`
#
expect_time=3
expect_top1=0.741
echo "Average inference time is $avg_time ms, expect time is <$expect_time ms"
echo "Top1 accuarcy is $top1, expect top1 is >$expect_top1"
if [[ $avg_time < $expect_time ]] && [[ $top1 > $expect_top1 ]] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi
