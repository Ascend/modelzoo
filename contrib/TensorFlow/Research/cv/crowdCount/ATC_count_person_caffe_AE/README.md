## 模型功能

密集人群计数。

## 原始模型

参考实现 ：


原始模型权重下载地址 :


https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.prototxt

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/insert_op.cfg




## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --input_shape="blob1:1,3,800,1408" --weight="count_person.caffe.caffemodel" --input_format=NCHW --output="count_person.caffe" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=0 --model="count_person.caffe.prototxt" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
 ./msame --model ../../../models/count_person.caffe.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 173.838970 ms
Inference average time without first time: 173.826838 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!

```

Batch: 1, shape: 1 * 3 * 800 * 1408，带AIPP，平均推理性能173ms

## 精度测试

待完善

推理效果
![输入图片说明](https://images.gitee.com/uploads/images/2021/0204/191142_a0256907_5578318.png "1612437078(1).png")
