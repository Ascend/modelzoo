## 模型功能

该模型用于图像文本检测。

## 原始模型

参考实现 ：

 https://github.com/HangZhouShuChengKeJi/chinese-ocr 

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/chinese-ocr/chineseocr.pb

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/chinese-ocr/chineseocr_aipp.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/chinese-ocr/chineseocr.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --model=./chineseocr.pb --framework=3 --output=./chineseocr --soc_version=Ascend310 --insert_op_conf=./chineseocr_aipp.cfg --input_shape="the_input:1,32,320,1"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model chineseocr.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 4.018600 ms
Inference average time without first time: 3.998333 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 32,320,1，带AIPP，平均推理性能3.998333 ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0111/120031_a9284a67_8113712.png "1.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0111/120043_5f11ce67_8113712.png "2.png")