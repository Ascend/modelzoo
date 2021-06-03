## 模型功能

 对图像中的物体进行分类。

## 原始模型

参考实现 ：

https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet50/resnet50.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet50/resnet50.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet50/resnet50.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
wget https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/resnet50/insert_op.cfg
```

```
atc --input_shape="data:1,3,224,224" --weight="resnet50.caffemodel" --input_format=NCHW --output="resnet50" --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=0 --model="resnet50.prototxt" --output_type=FP32
```

## 使用msame工具推理

参考 https://github.com/Ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
 ./msame --model ../../../models/resnet50.om --output output/ --loop 100

```

性能测试数据为：

```
[INFO] output data success
Inference average time: 2.788920 ms
Inference average time without first time: 2.784808 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape:  224 *224 *3，带有AIPP，平均推理性能 2.78ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1218/101122_8d4dad5f_5578318.jpeg "out_dog2_1024_683.jpg")