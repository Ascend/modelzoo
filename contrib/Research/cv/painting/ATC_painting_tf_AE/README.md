## 模型功能

该模型用于生成风景画。

## 原始模型

参考实现 ：

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --output_type=FP32 --input_shape="objs:9;coarse_layout:1,256,256,17"  --input_format=NHWC --output="AIPainting_v2" --soc_version=Ascend310 --framework=3  --model="AIPainting_v2.pb"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/AIPainting_v2.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 195.832410 ms
Inference average time without first time: 195.707606 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

 Batch：1,shape1: 9 shape2: 1 * 256 * 256 * 17,不带AIPP ，平均推理性能195.71ms

## 精度测试

待完善

推理效果

![大海](https://images.gitee.com/uploads/images/2020/1127/161214_f419d7b2_7990837.jpeg "80e3f5c9dc91af8dc84eb7ed1063c24.jpg")

​							                                                     图为AI生成的大海

