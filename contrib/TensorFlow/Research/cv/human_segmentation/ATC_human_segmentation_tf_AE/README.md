## 模型功能

该模型用于人体的语义分割。

## 原始模型

参考实现 ：


原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/insert_op.cfg

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/human512_3c_binary_512x512.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/human512_3c_binary_512x512.om

使用ATC模型转换工具进行模型转换时可以参考如下指令：

```
atc --input_shape="input_rgb:1,512,512,3" --input_format=NHWC --output=human512_3c_binary_512x512 --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=3 --model=./human512_3c_binary_512x512.pb
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model human512_3c_binary_512x512.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 115.150800 ms
Inference average time without first time: 115.068333 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 512 * 512 * 3，带AIPP，平均推理性能 115.150800 ms

## 精度测试

待完善

推理效果
![输入图片说明](https://images.gitee.com/uploads/images/2021/0208/142001_cf375ae5_5302634.jpeg "source.jpg")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0208/142016_523427e1_5302634.png "target.png")
