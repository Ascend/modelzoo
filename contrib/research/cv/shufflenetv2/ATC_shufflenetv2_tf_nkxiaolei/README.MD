## 1、原始模型
模型下载来自：https://github.com/opconty/keras-shufflenetV2 \
将保存的h5文件转化为pb文件之后，使用ATC工具转化成om模型

## 2、转om模型
obs链接：obs://nkxiaolei88/ATC Model/ShufflenetV2/shufflenetv2_emotion_recogn.om

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="Input:1,48,48,1" --check_report=/root/modelzoo/shufflenetv2_emotion_recogn/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/shufflenetv2_emotion_recogn/device/shufflenetv2_emotion_recogn" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/shufflenetv2_emotion_recogn.pb" 
``` 

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/shufflenetv2_emotion_recogn.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 0.888980 ms
Inference average time without first time: 0.885677 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x48x48x1，不带AIPP，平均推理性能0.886ms

## 6、精度测试：
这个网络是使用ShufflenetV2来实现人脸表情识别的，输入为48*48的灰度图，输出为表情分类：\
["angry", "scared", "happy", "sad", "surprised", "neutral"]\
使用的数据集为icml_face_data, 数据集的下载和预处理参考：\
https://www.kaggle.com/mohammed94/facial-emotion-detection-cnn

原始的数据集为csv文件，将灰度图的信息和标签保存在csv文件中，本案例中已经集成预处理、推理和精度计算：\
先解压出public_test.txt
```
cd ground_truth/
unzip public_test.zip
```

bash start_inference.sh
```
003583, inference label:2, gt_label:2
003584, inference label:3, gt_label:3
003585, inference label:2, gt_label:2
003586, inference label:3, gt_label:3
003587, inference label:3, gt_label:3
003588, inference label:5, gt_label:3
Totol pic num: 3589, Top1 accuarcy: 0.5575
```
Public_test数据集中3589张图片，推理的精度为0.5575，而该模型在GPU上训练精度为0.55