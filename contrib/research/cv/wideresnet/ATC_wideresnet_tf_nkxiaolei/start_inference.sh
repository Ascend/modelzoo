#! /bin/bash
input="input/"
output="output/"
label="ground_truth/cifar10_val_1w_labels.txt"
model="model/WRN-28-10.om"

rm -rf $output/*
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3.7.5 accuarcy_top1.py $output $label
