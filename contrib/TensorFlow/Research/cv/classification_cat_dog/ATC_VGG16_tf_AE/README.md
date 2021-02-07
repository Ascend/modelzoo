## 模型功能

 实现猫够分类。

## 原始模型

原始模型参考链接：

https://github.com/keras-team/keras/tree/master/keras/applications

原始模型下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vgg16_cat_dog/vgg16_cat_dog.pb

 对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vgg16_cat_dog/insert_op.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/vgg16_cat_dog/vgg16_cat_dog.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --output_type=FP32 --input_shape="input_1:1,224,224,3"  --input_format=NHWC --output="vgg16_cat_dog" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=3 --model="./vgg16_cat_dog.pb" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在masme可执行文件同级目录下，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model vgg16_cat_dog.om --output output --outfmt TXT --loop 10
```

```
[INFO] output data success
Inference average time: 4.298200 ms
Inference average time without first time: 4.287111 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 224* 224*3 ，带AIPP，平均推理性能4.287111

## 精度测试

待完善

推理效果

