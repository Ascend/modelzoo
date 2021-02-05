## 模型功能

通过读取本地雨天退化图像数据，对场景中的雨线、雨雾进行去除，实现图像增强效果。

## 原始模型

参考实现 ：

https://gitee.com/caizhi_969/DeRain.git

原始模型下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/DeRain/insert_op.cfg

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/DeRain/frozen_graph_noDWT_V2.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/DeRain/DeRain.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./frozen_graph_noDWT_V2.pb --input_shape="degradated_image:1,256,256,1" --framework=3 --output=./DeRain  --soc_version=Ascend310   --insert_op_conf=./insert_op.cfg
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model model/DeRain.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 22.149870 ms
Inference average time without first time: 22.147222 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 256 * 256，不带AIPP，平均推理性能22.15ms

## 精度测试

待完善

推理效果


![输入图片说明](https://images.gitee.com/uploads/images/2021/0205/145926_8850fbc8_8018002.png "005_in.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0205/145958_0e01a782_8018002.png "out_005_in.png")