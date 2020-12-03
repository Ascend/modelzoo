## 1、原始模型
https://www.huaweicloud.com/ascend/resources/modelzoo/Models/f3f2b8541b8a48f8b3c996f33d46f041, 下载对应的ckpt，转成pb模型

## 2、转om模型
obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/Xception/xception_tf.om

## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/xception_tf.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 4.218640 ms
Inference average time without first time: 4.214990 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x224x224x3，不带AIPP，平均推理性能4.21ms

## 6、精度测试：

### 6.1 下载ImageNet 2012 val数据集

### 6.2 图片预处理
下载好的数据集JPEG图片放在任意其他目录(以originnal_pic为例)，执行预处理脚本生成bin文件： 
```
python3.7.5 img_preprocess.py ./originnal_pic/ 
```
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

### 6.3 执行推理和精度计算的shell脚本： 
./start_inference.sh
```
ILSVRC2012_val_00049998.JPEG, inference label:232, gt_label:232
ILSVRC2012_val_00049999.JPEG, inference label:982, gt_label:982
ILSVRC2012_val_00050000.JPEG, inference label:355, gt_label:355
```
50000张Val数据集在NPU上推理的top1精度为：0.730， 论文参考精度值：0.79

因为使用的图片分辨率较低，输出的精度结果比预期低一点。修改模型的输入shape为：1x299x299x3，重新生成om模型后，重复6.1~6.3步骤：

50000张Val数据集在NPU上推理的top1精度为：0.784， 论文参考精度值：0.79，性能会差一点，单帧耗时14.9 ms
