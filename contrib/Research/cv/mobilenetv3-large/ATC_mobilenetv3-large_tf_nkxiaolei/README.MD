## 1、原始模型
链接：https://github.com/xiaochus/MobileNetV3, 下载对应的ckpt，转成pb模型

## 2、转om模型
obs链接：obs://nkxiaolei88/ATC Model/MobileNetV3-large/mobilenet-v3-large.om

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="input:1,224,224,3" --check_report=/root/modelzoo/mobilenet-v3-large/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/mobilenet-v3-large/device/mobilenet-v3-large" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/mobilenet-v3-large.pb" 
```

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/mobilenet-v3-large.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 3.100290 ms
Inference average time without first time: 3.096515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x224x224x3，不带AIPP，平均推理性能3.09ms

## 6、精度测试：

### 6.1 下载ImageNet 2012 val数据集：
http://www.image-net.org/download-images

### 6.2 图片预处理
下载好的数据集JPEG图片放在任意其他目录，执行预处理脚本生成bin文件：
```
python3.7.5 img_preprocess.py ./originnal_pic/
```
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

### 6.3 执行推理和精度计算的shell脚本：
bash start_inference.sh
```
ILSVRC2012_val_00049998.JPEG, inference label:232, gt_label:232
ILSVRC2012_val_00049999.JPEG, inference label:982, gt_label:982
ILSVRC2012_val_00050000.JPEG, inference label:355, gt_label:355
```

50000张Val数据集在NPU上推理的top1精度为：0.755， 论文参考精度值：0.752