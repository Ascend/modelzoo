1、转换模型，下载地址来自于Github:
https://github.com/hunglc007/tensorflow-yolov4-tflite
原图是用Keras训练得到的，先转成h5文件，再将h5转成pb文件。
后处理中的where算子不支持，因此拆分掉后处理部分的算子。
拆分pb模型的命令：
python3.7.5 -m tensorflow.python.tools.optimize_for_inference --input=yolov4.pb --output=yolov4_no_postprocess.pb --input_names="Input" --output_names="functional_1/tf_op_layer_concat_13/concat_13,functional_1/tf_op_layer_concat_14/concat_14"

2、下载验证数据集： COCO 2017 Dataset

3、将JPG文件转换成二进制bin文件

4、使用msame工具进行推理
下载om模型，obs链接：obs://nkxiaolei88/ATC Model/YoloV4/yolov4_no_postprocess.om 
将下载的om文件放在model文件夹 使用nsame推理工具，参考如下命令，发起推理性能测试： 
./msame --model model/yolov4_no_postprocess.om --output output/ --loop 100 

[INFO] output data success
Inference average time: 44.473680 ms
Inference average time without first time: 44.468061 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x416x416x3，不带AIPP，平均推理性能44.47ms

5、根据输出的bin文件计算精度。