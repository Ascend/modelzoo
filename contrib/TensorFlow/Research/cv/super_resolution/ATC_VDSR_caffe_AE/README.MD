## 模型功能

模糊图像变清晰。

## 原始模型

参考实现 ：

https://github.com/wangxuewen99/Super-Resolution/tree/master/VDSR

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/VDSR/VDSR.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/VDSR/VDSR.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/VDSR/VDSR_840_840.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./VDSR.prototxt --weight=./VDSR.caffemodel --framework=0 --input_format=NCHW --input_shape="data: 1, 1, 840, 840" --output=./VDSR_840_840 --soc_version=Ascend310 --output_type=FP32
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model model/VDSR_840_840.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 170.358490 ms
Inference average time without first time: 170.370444 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 840 * 840，不带AIPP，平均推理性能170.35ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/154713_d1cf11df_8113712.png "图片1.png")
![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/154747_583198c2_8113712.png "图片2.png")
