#! /bin/bash
input="./input/"
output="output/"
label="./ground_truth/img_label.txt"
model="model/shufflenetv2.om"

rm -rf $output/*
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3 accuracy_top1.py $output $label
