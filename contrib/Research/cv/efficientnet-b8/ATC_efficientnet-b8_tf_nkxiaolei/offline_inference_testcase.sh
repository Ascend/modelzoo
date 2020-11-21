#! /bin/bash
#Ascend社区已预置的数据集、预训练模型、ATC-OM模型等
originnal_pic="/data/ModelZoo/Dataset/ILSVRC2012_val/"
ori_model="/data/ModelZoo/Models/nkxiaolei/EfficientNet-B8/efficientnet-b8.om"
input="input/"
output="output/"
label="ground_truth/val_map.txt"
model=ori_model
#开发个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载

#case主体，开发者根据不同模型写作
python3.7.5 img_preprocess.py --src_path=$originnal_pic --dst_path=$input --pic_num=100

#start inference and postprocess
python3 testcase.py $model $input $output $label > result.log

#结构判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
res=`grep "key" result.log`
