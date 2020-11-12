#! /bin/bash
input="input/"
output="output/"
label="ground_truth/fasttext_label.bin"
model="model/fast_text_frozen.om"

rm -rf $output/
mkdir $output
#start infence
./msame --model $model --input $input --output $output

#top1 accuarcy
python3.7.5 accuarcy_top1.py $output $label
