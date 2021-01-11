## 模型功能

该模型用于人像分割及背景替换。

## 原始模型

参考实现 ：

https://github.com/dong-x16/PortraitNet 

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/PortraitNet%20/portrait.pb

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/PortraitNet%20/insert_op.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/PortraitNet%20/portrait.om

使用ATC模型转换工具进行模型转换时可以参考如下指令：

```
atc --model=portrait.pb  --input_shape="Inputs/x_input:1,224,224,3"  --framework=3  --output=portrait --insert_op_conf=insert_op.cfg --soc_version=Ascend310 
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
Inference average time: 6.355180 ms
Inference average time without first time: 6.353273 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 224 * 224 * 3，带AIPP，平均推理性能 6.353273 ms

## 精度测试

待完善

推理效果


<div style="float:left;border:solid 1px 000;margin:2px;"><img src="C:\Users\83395\Desktop\新建文件夹\seg\ori.jpg"  width="200" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="C:\Users\83395\Desktop\新建文件夹\seg\background.jpg"  width="200" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="C:\Users\83395\Desktop\新建文件夹\seg\new.jpg"  width="200" ></div>



