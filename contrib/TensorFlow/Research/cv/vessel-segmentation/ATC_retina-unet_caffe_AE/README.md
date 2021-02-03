## 模型功能

该模型用于对眼底图像进行血管分割。

## 原始模型

参考实现 ：

 https://github.com/orobix/retina-unet#retina-blood-vessel-segmentation-with-a-convolution-neural-network-u-net
 
原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/retina-unet/deploy_vel_ascend.prototxt

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/retina-unet/vel_hw_iter_5000.caffemodel


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/retina-unet/retina-unet.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --model=caffe_model/deploy_vel_ascend.prototxt --weight=caffe_model/vel_hw_iter_5000.caffemodel --framework=0 --output=model/retina-unet --soc_version=Ascend310 --input_format=NCHW --input_fp16_nodes=data -output_type=FP32
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model chineseocr.om  --output output/ --loop 10
./msame --model /home/HwHiAiUser/bobeexu/vessel2/model/retina-unet.om  --output /home/HwHiAiUser/bobeexu/out --outfmt TXT --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 13.405190 ms
Inference average time without first time: 13.404515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 3,512,512，平均推理性能13.405190 ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0203/163317_ba680af8_1656526.jpeg "result.jpg")

