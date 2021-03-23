## 模型功能

该模型实现了对图像hdr增强的功能

## 原始模型

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/image_HDR_enhance/image_HDR_enhance.pb

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/image_HDR_enhance/image_HDR_enhance.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --model=./image_HDR_enhance.pb --framework=3 --output=./image_HDR_enhance --soc_version=Ascend310 --input_shape="input:1,512,512,3" --input_format=NHWC --output_type=FP32
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model image_HDR_enhance.om  --output output/ --loop 10
```

性能测试数据为：

```
loop:10
******************************
Test Start!
[INFO] acl init success
[sudo] password for HwHiAiUser: 
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] malloc buffer for mem , require size is 217977856
[INFO] malloc buffer for weight,  require size is 829952
[INFO] load model /home/HwHiAiUser/caoliang/hdr_200dk/model/model.om success
[INFO] create model description success
[INFO] create model output success
[INFO] model execute success
Inference time: 92.675ms
[INFO] model execute success
Inference time: 92.618ms
[INFO] model execute success
Inference time: 92.595ms
[INFO] model execute success
Inference time: 92.627ms
[INFO] model execute success
Inference time: 92.524ms
[INFO] model execute success
Inference time: 92.43ms
[INFO] model execute success
Inference time: 92.703ms
[INFO] model execute success
Inference time: 92.818ms
[INFO] model execute success
Inference time: 92.622ms
[INFO] model execute success
Inference time: 92.529ms
output//20210203_072604
[INFO] output data success
Inference average time: 92.614100 ms
Inference average time without first time: 92.607333 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

Batch: 1, shape: 3,512,512 不带AIPP，平均推理性能92.614100 ms


### 推理效果

推理前：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0203/151356_05b30074_8083019.png "a4962.png")

推理后：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0203/151626_0c89672a_8083019.png "a4962.png")
