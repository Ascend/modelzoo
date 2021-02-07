## 模型功能

该模型实现蛋白质亚细胞定位预测

## 原始模型

参考实现 ：

 

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/hpa/hpa.prototxt    

原始模型权重文件下载地址

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/hpa/hpa.caffemodel    

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/hpa/deploy_vel.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --model=./hpa.prototxt --weight=./hpa.caffemodel --framework=0 --output=./deploy_vel  --soc_version=Ascend310 --input_format=NCHW --input_fp16_nodes=data --output_type=FP32 --out_nodes="score:0"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model rcf.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 1.208800 ms
Inference average time without first time: 1.184444 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

平均推理性能1.184444 ms

## 精度测试

待完善
推理效果   
![输入图片说明](https://images.gitee.com/uploads/images/2021/0205/150818_93dfa9dc_7985487.png "屏幕截图.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0205/150732_31e260fc_7985487.png "屏幕截图.png")