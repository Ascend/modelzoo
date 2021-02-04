## 模型功能

该模型用于模糊图像变清晰。

## 原始模型

参考实现 ：

无

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/DeblurGAN/DeblurrGAN-pad-01051648.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/DeblurGAN/DeblurrGAN_pad_1280_720.om

使用ATC模型转换工具进行模型转换时可以参考如下指令：

```
atc --input_shape="blur:1,720,1280,3" --input_format=NHWC --output="./DeblurrGAN_pad_1280_720" --soc_version=Ascend310 --framework=3 --model="./DeblurrGAN-pad-01051648.pb" --log=info
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model portrait.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 683.453900 ms
Inference average time without first time: 683.422111 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1280 * 720 * 3，不带AIPP，平均推理性能 683.422111 ms

## 精度测试

待完善

推理效果
