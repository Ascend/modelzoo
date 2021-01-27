## 1、原始模型
https://github.com/qubvel/efficientnet, 下载对应的ckpt，转成pb模型

## 2、转om模型
obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/EfficientNet-B8/efficientnet-b8.om

atc转换命令参考：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="images:1,672,672,3" --check_report=/root/modelzoo/efficientnet-b8/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/efficientnet-b8/device/efficientnet-b8" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/efficientnet-b8.pb" 
```

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/efficientnet-b8.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 139.019250 ms
Inference average time without first time: 139.011384 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x672x672x3，不带AIPP，平均推理性能139ms

## 6、精度&性能自测试用例
自测试使用Imagnet2012 Val中前100张图片来验证性能和精度，\
然后根据获取到的回显来解析出平均推理时间和精度，判断用例是否通过

执行入口：
```
bash offline_inference_testcase.sh
```
100张图片测试结果：
```
Average inference time is 138.61 ms, expect time is <145 ms
Top1 accuarcy is 0.8700, expect top1 is >0.86
Run testcase success!
```

## 7、全量数据集精度测试：

### 7.1 下载ImageNet 2012 val数据集

### 7.2 预处理

下载好的数据集JPEG图片放在任意其他目录(以originnal_pic为例)，执行预处理脚本生成bin文件：
```
python3.7.5 img_preprocess.py --src_path=$originnal_pic --dst_path=$input
```
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

### 7.3 执行推理和精度计算
执行shell脚本： ./start_inference.sh
```
ILSVRC2012_val_00049998.JPEG, inference label:232, gt_label:232
ILSVRC2012_val_00049999.JPEG, inference label:982, gt_label:982
ILSVRC2012_val_00050000.JPEG, inference label:355, gt_label:355
```
50000张Val数据集在NPU上推理的top1精度为：0.852， 论文参考精度值：0.854