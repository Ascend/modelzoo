## 1、原始模型
参考https://github.com/statech/resnet

使用Keras创建Resnet152模型，然后将Keras模型转成tf的frozen pb

## 2、转om模型
obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/Resnet_152/resnet152_tf.om

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="data:1,224,224,3" --check_report=/root/modelzoo/resnet152_tf/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/resnet152_tf/device/resnet152_tf" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/resnet152_tf.pb"
```

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/resnet152_tf.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 11.337590 ms
Inference average time without first time: 11.331727 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x224x224x3，不带AIPP，平均推理性能11.33ms

## 6、精度测试：

### 6.1 下载ImageNet 2012 val数据集

### 6.2 图片预处理
下载好的数据集JPEG图片放在任意其他目录(以originnal_pic为例)，执行预处理脚本生成bin文件：
```
python3.7.5 img_preprocess.py ./originnal_pic/ ./input/
```
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

### 6.3 执行推理和精度计算的shell脚本： 
./start_inference.sh
```
ILSVRC2012_val_00049987.JPEG, inference label:44, gt_label:44
ILSVRC2012_val_00049988.JPEG, inference label:81, gt_label:81
ILSVRC2012_val_00049989.JPEG, inference label:399, gt_label:399
ILSVRC2012_val_00049990.JPEG, inference label:24, gt_label:24
ILSVRC2012_val_00049991.JPEG, inference label:120, gt_label:120
ILSVRC2012_val_00049992.JPEG, inference label:357, gt_label:357
```
50000张Val数据集在NPU上推理的top1精度为：0.4569， 论文参考精度值：0.7857\
下载的预训练模型精度不高