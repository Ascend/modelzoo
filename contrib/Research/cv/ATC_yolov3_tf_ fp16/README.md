## 模型功能

 对图像中的物体进行识别分类。

## 原始模型

参考实现 ：

https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/detection/YoloV3_for_TensorFlow


原始bp模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_tf/yolov3_tf.pb

AIPP下载地址 ：

https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/YOLOV3_VOC_detection_picture/insert_op.cfg

## om模型

om模型下载地址：

20.0版本模型：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_tf/yolov3_ascend310_aipp_1_batch_1_input_fp16_output_FP32_C73.om

20.1版本模型：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/yolov3_tf/yolov3_ascend310_aipp_1_batch_1_input_fp16_output_FP32_C75.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --output_type=FP32 --input_shape="input/input_data:1,416,416,3" --input_fp16_nodes="" --input_format=NHWC --output=yolov3_ascend310_aipp_1_batch_1_input_fp16_output_FP16_C73 --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=3 --save_original_model=false --model=./yolov3_tf.pb
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/yolov3_ascend310_aipp_1_batch_1_input_fp16_output_FP16_C73.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100

```

```
[INFO] output data success
Inference average time: 14.976350 ms
Inference average time without first time: 14.975758 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 416 * 416 * 3，带AIPP，平均推理性能 14.976ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160255_32f676b5_8113712.png "图片4.png")