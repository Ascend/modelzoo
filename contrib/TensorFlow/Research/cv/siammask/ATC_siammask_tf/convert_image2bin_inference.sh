# echo "begin convert image to bin"
tools_path="./tools"
project_path="./"

rm -rf tmp/output1

python3.7  $tools_path/img2bin/img2bin.py -i $project_path/tmp/search.jpg -w 255 -h 255  -f RGB -a NHWC -t  float32 -o $project_path/tmp/
python3.7  $tools_path/img2bin/img2bin.py -i $project_path/tmp/template.jpg -w 127 -h 127  -f RGB -a NHWC -t  float32 -o $project_path/tmp/

# echo "end convert image to bin"

$tools_path/msame/out/msame --model ./model/siammask.om --input $project_path/tmp/template.bin,$project_path/tmp/search.bin --output $project_path/tmp/output1 --outfmt BIN --loop 1

# echo "sucess inference"