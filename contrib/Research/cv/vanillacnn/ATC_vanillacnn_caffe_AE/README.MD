## 模型功能

 用于人脸关键点检测，识别人脸中左眼、右眼、鼻子、左嘴角、右嘴角。

## 原始模型

参考实现 ：

https://github.com/ishay2b/VanillaCNN

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vanillacnn/vanillacnn.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vanillacnn/vanilla_deploy.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vanillacnn/vanillacnn_framework_caffe_aipp_0_batch_4_input_fp32_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model=./vanilla_deploy.prototxt --weight=./vanillacnn.caffemodel --framework=0 --output=./vanillacnn_framework_caffe_aipp_0_batch_4_input_fp32_output_FP32 --soc_version=Ascend310
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/vanillacnn_framework_caffe_aipp_0_batch_4_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 0.524740 ms
Inference average time without first time: 0.522434 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 4, shape: 3 * 40 * 40，不带AIPP，平均推理性能0.52ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160149_83c721c6_8113712.jpeg "e269decb-5f25-4d8c-e4df-3fb98da9d306.jpg")