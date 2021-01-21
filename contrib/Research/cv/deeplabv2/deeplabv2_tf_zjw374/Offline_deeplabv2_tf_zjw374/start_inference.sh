#! /bin/bash
input="bin_dataset/JPEGImages/"
output="output/"
label="bin_dataset/SegmentationClassAug"
model="model/deeplabv2-2000.om"

#rm -rf $output/*
#start infence
#./msame --model $model --input $input --output $output

#top1 accuarcy
python3 310Inference.py
