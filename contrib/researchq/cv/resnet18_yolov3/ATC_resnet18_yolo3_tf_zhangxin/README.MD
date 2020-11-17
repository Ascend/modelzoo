## 1、原始模型
用户如果想自行转换om文件，可下载原生Tensorflow模型（.pb）及AIPP配置文件，若不想自行转换，可直接跳到 [2、转om模型](#2、转om模型) 


obs链接:  https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3-RESNET18%20/ATC%20YOLOV3-RESNET18%20%28UINT8%29%20from%20TensorFlow%20-%20Ascend%20310.zip 

下载后，使用ATC模型转换工具或者执行转换命令进行转换：
```
atc --model="./yolo3_resnet18.pb" --output_type=FP32 --input_shape="images:1,352,640,3" --insert_op_conf=./insert_op.cfg --input_format=NHWC --output="./yolov3resnet18_framework_tensorflow_aipp_1_batch_1_input_uint8_output_FP32" --framework=3  --soc_version=Ascend310
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3-RESNET18%20/yolov3resnet18_framework_tensorflow_aipp_1_batch_1_input_uint8_output_FP32.om 


## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/yolov3resnet18_framework_tensorflow_aipp_1_batch_1_input_uint8_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100

```
```
[INFO] output data success
Inference average time: 4.538660 ms
Inference average time without first time: 4.535606 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batch: 1, shape: 352 * 640 * 3，带AIPP，平均推理性能4.54ms

## 6、精度测试
待完善

推理效果

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml3032\wps4.jpg)