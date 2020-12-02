## 模型功能

识别图像中的人脸。

## 原始模型

参考实现 ：

https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/facedection/face_detection_fp32.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/facedection/face_detection.prototxt

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/facedection/insert_op.cfg




## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/facedection/ssd-facedection_framework_caffe_aipp_1_batch_1_input_fp16_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model=./face_detection.prototxt --weight=./face_detection_fp32.caffemodel --framework=0 --output=./face_detection --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/face_detection_framework_caffe_aipp_1_batch_1_input_fp32_output_FP16.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 10.470210 ms
Inference average time without first time: 10.466707 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 300 * 304，带AIPP，平均推理性能10.47ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/154908_240491cd_8113712.png "08281691-0233-4419-f611-2cf65624d5d8.png")
