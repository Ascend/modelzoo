#! /bin/bash
originnal_pic="/data/ModelZoo/Dataset/ILSVRC2012_val/"
ori_model="/data/ModelZoo/Models/nkxiaolei/EfficientNet-B8/efficientnet-b8.om"
input="input/"
output="output/"
label="ground_truth/val_map.txt"
model="model/efficientnet-b8.om"

rm -rf $input
mkdir $input
python3.7.5 img_preprocess.py --src_path=$originnal_pic --dst_path=$input --pic_num=100
cp $ori_model $model
rm -rf $output/*
#start inference and postprocess
python3 testcase.py $model $input $output $label
