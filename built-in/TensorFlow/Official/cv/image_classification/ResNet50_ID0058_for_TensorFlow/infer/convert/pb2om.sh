model_path=$1
framework=$2
output_model_name=$3

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--framework=$framework \
--output=$output_model_name \
--input_format=NHWC --input_shape="Placeholder:1,224,224,3" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=./aipp_resnet50.aippconfig