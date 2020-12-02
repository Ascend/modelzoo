## 模型功能

该模型用于识别图片中垃圾的种类。

## 原始模型

参考实现 ：

https://github.com/shicai/MobileNet-Caffe

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/garbage/mobilenetv2.pb

对应的cfg文件：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/garbage/insert_op.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/garbage/mobilenetv2.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model=mobilenetv2.pb  --framework=1 --output=mobilenetv2 --insert_op_conf=mobilenetv2.cfg  --soc_version=Ascend310
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/mobilenetv2.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 1.540100 ms
Inference average time without first time: 1.537051 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1 , shape: 224 * 224 * 3，带AIPP，平均推理性能1.54 ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1127/160833_788a5493_7990837.jpeg "cloths.jpg")

![输入图片说明](https://images.gitee.com/uploads/images/2020/1127/160849_2a8d7431_7990837.jpeg "lump.jpg")

![输入图片说明](https://images.gitee.com/uploads/images/2020/1127/160859_1457cc74_7990837.jpeg "newspapper.jpg")
