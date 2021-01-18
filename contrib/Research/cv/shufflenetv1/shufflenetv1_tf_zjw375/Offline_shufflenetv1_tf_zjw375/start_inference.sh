#! /bin/bash
input="./input/"
output="./output/"
label="./label_output"
model="model/shufflenetv1.om"

rm -rf $output/*
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3.7.5 accuracy_top1.py $output $label
