## 模型功能

该模型用于智能小车检测车道线，实现循道行驶。

## 原始模型

参考实现 ：

https://github.com/weiliu89/caffe

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/car/following/road_following_model.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/car/following/road_following_model.prototxt

对应的cfg文件下载地址： 

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/car/following/insert_op_road_following.cfg

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/car/following/road_following_model.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model="road_following_model.prototxt" --weight="road_following_model.caffemodel" --soc_version=Ascend310 --framework=0 --output="road_following_model" --insert_op_conf=insert_op_road_following.cfg 
```



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
 ./msame --model ../model/road_following_model.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 1.583040 ms
Inference average time without first time: 1.578444 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 224 * 224 * 3，带AIPP，平均推理性能 1.58ms

## 精度测试

待完善


