#! /bin/bash
input="input/"
output="output/"
label="ground_truth/shufflenetv2_emotion_labels.txt"
model="model/shufflenetv2_emotion_recogn.om"

rm -rf $input
mkdir $input
python3.7.5 read_csv_to_bin.py
rm -rf $output/*
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3.7.5 accuarcy_top1.py $output $label
