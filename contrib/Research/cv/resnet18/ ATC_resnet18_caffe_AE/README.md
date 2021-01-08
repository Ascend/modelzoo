## 模型功能

检测输入图片中的文字并显示。

## 原始模型

参考实现 ：

https://gitee.com/yaoyaoling11/hand-write-c73

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/resnet.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/resnet.prototxt

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/insert_op.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/resnet_framwork_caffe_aipp_1_batch_1_input_fp32_output_fp32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model=./resnet.prototxt --weight=./resnet.caffemodel --framework=0 --output=resnet_framwork_caffe_aipp_1_batch_1_input_fp32_output_fp32 --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --input_shape="data:1,3,112,112" --input_format=NCHW
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/resnet_framwork_caffe_aipp_1_batch_1_input_fp32_output_fp32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 4.538660 ms
Inference average time without first time: 4.535606 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 352 * 640 * 3，带AIPP，平均推理性能4.54ms

## 精度测试

待完善

推理效果

