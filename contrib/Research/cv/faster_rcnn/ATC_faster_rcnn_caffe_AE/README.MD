## 模型功能

 对图像中的物体进行识别分类。

## 原始模型

参考实现 ：

https://github.com/rbgirshick/py-faster-rcnn/

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/faster_rcnn/faster_rcnn.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/faster_rcnn/faster_rcnn.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/faster_rcnn/faster_rcnn_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=faster_rcnn.prototxt --weight=faster_rcnn.caffemodel --framework=0 --output=faster_rcnn_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32.om --soc_version=Ascend310 --input_shape="data:1,3,-1,-1;im_info:1,3" --dynamic_image_size="512,512;600,600;800,800"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/faster_rcnn_framework_caffe_ascend310_aipp_1_batch_1_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 24.962560 ms
Inference average time without first time: 24.958515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 3 * 300 * 300，不带AIPP，平均推理性能 24.96ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/155216_6451a809_8113712.png "bicycle.png")
