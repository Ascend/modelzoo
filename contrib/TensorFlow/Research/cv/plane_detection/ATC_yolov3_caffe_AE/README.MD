
## 模型功能

 实现对遥感图像中飞机目标的检测。

## 原始模型

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_plane/yolov3.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_plane/yolov3.prototxt

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_plane/aipp_nv12.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_plane/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=yolov3.prototxt --weight=yolov3.caffemodel --framework=0 --output=yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om --soc_version=Ascend310 --insert_op_conf=aipp_nv12.cfg
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../../../ATC_YOLOv3_caffe/model/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om  --output ../../../ATC_YOLOv3_caffe/msame_infer --outfmt TXT --loop 100
```

```
Inference average time: 9.017290 ms
Inference average time without first time: 9.010545 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!

```

Batch: 1, shape: 416* 416 *3 ，带AIPP，平均推理性能9.01ms

## 精度测试

待完善

推理效果

![输入图片说明](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_plane/verify_image/verify_test.jpg "yolov3-caffe.jpg")

。