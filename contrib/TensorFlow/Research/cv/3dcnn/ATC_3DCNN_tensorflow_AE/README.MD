## 模型功能

 对视频中的动作进行识别分类。

## 原始模型

参考实现 ：

https://gitee.com/azureluck/gesture_recognition 

原始下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/3D_gesture_recognition/3d_gesture_recognition.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/3D_gesture_recognition/3d_gesture_recognition.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=3d_gesture_recognition.pb  --framework=3 --output=3d_gesture_recognition --soc_version=Ascend310 --input_shape="X:1,16,112,112,3" --input_format=NDHWC      
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/3d_gesture_recognition.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 17.811650 ms
Inference average time without first time: 17.810657 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 16 * 112 * 112 * 3，不带AIPP，平均推理性能 18.81ms

## 精度测试

待完善

