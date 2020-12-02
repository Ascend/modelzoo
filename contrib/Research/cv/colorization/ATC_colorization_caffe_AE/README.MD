## 模型功能

通过对黑白图像进行着色变为彩色图像。

## 原始模型

参考实现 ：

https://github.com/richzhang/colorization

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization_framework_caffe_aipp_0_batch_1_input_fp32_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --output_type=FP32 --input_shape="data_l:1,1,224,224" --weight="./colorization.caffemodel"  --input_format=NCHW --output="./colorization" --soc_version=Ascend310 --framework=0 --save_original_model=false --model="./colorization.prototxt" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model model/colorization_framework_caffe_aipp_0_batch_1_input_fp32_output_FP32.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 10.212930 ms
Inference average time without first time: 10.209758 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 224 * 224，不带AIPP，平均推理性能10.21ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/154713_d1cf11df_8113712.png "图片1.png")
![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/154747_583198c2_8113712.png "图片2.png")