
project_path=$(dirname $(readlink -f "$0"))

python3.7  img2bin.py -i $1 -w 800 -h 800 -f BGR -a NHWC -t float32 \
                            -m [122.67891434,116.66876762,104.00698793] \
                            -c [0.00392156862745098039,0.00392156862745098039,0.00392156862745098039] \
                            -o $project_path/tmp

output="output/"
model="model/db_resnet.om"

rm -rf $output/*

./msame --model $model --input tmp/$2.bin --output $output