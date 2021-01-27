## 1、原始模型
https://github.com/asmith26/wide_resnets_keras, 下载代码和数据集，进行训练。

## 2、转om模型
obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/Wide-Resnet/WRN-28-10.om

ATC转换命令:
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="x:1,32,32,3" --check_report=/root/modelzoo/wide_resnet/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/wide_resnet/device/wide_resnet" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/wide_resnet.pb" 
```

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/WRN-28-10.om --output output/ --loop 100

[INFO] output data success

Inference average time: 3.474360 ms

Inference average time without first time: 3.470899 ms

[INFO] unload model success, model Id is 1

[INFO] Execute sample success.


1Batch，shape:1x32x32x3，不带AIPP，平均推理性能3.47ms

## 6、精度测试：
使用Cifar10验证集，1万张，分辨率为32x32x3\
下载数据集：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/Wide-Resnet/input.tar.gz 至当前目录\
下载标签：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/Wide-Resnet/cifar10_val_1w_labels.txt 至ground_truth目录中

### 6.1 解压数据集，已经做好了预处理的图片，并生成为bin文件：
```
tar -zxvf input.tar.gz
```
### 6.2 开始执行精度测试：
```
bash start_inference.sh
```
得到的精度数据为：
```
009990, inference label:5, gt_label:5
009991, inference label:2, gt_label:2
009992, inference label:8, gt_label:8
009993, inference label:2, gt_label:2
009994, inference label:8, gt_label:8
009995, inference label:5, gt_label:5
009996, inference label:3, gt_label:3
009997, inference label:2, gt_label:2
009998, inference label:8, gt_label:8
009999, inference label:3, gt_label:3
Totol pic num: 10000, Top1 accuarcy: 0.9388
```
与GPU环境执行得到的精度数据一致：
```
79/78 [==============================] - 515s 7s/step - loss: 0.3046 - accuracy: 0.9387
INFO:root:Results:
INFO:root:Test loss: 0.304599791765213
INFO:root:Test accuracy: 0.9387000203132629
```
