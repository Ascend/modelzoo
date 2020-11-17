#! /bin/bash
input="input/"
output="output/"
label="ground_truth/val_map.txt"
model="model/noisy_student_efficientnet-l2.om"

rm -rf $output/*
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3.7.5 accuarcy_top1.py $output $label