if [ $# != 3 ]; then
  echo "USAGE: $0 xx.onnx xx batch"
  exit 1;
fi

bs=$3
atc --model=$1 --framework=5 --output=$2_bs$bs --input_format=NCHW --log=error --soc_version=Ascend310 --input_shape="images:$bs,3,640,640;img_info:$bs,4" --enable_small_channel=1 --input_fp16_nodes="images;img_info" --output_type=FP16

