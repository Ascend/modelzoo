## 模型功能

该模型用于将普通照片转换为卡通风格的图片。

## 原始模型

参考实现 ：

https://github.com/taki0112/CartoonGAN-Tensorflow 

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/cartoonization/cartoonization.pb

对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/cartoonization/insert_op.cfg


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/cartoonization/cartoonization.om

使用ATC模型转换工具进行模型转换时可以参考如下指令：

```
atc --output_type=FP32 --input_shape="train_real_A:1,256,256,3"  --input_format=NHWC --output="cartoonization" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=3 --model="cartoonization.pb" --precision_mode=allow_fp32_to_fp16
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/cartoonization.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 1065.778410 ms
Inference average time without first time: 1065.429838 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 256 * 256 * 3，带AIPP，平均推理性能 1065.4ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1127/160621_cdd46d90_7990837.png "cartoon4.png")

![输入图片说明](https://images.gitee.com/uploads/images/2020/1127/160650_cc926bb7_7990837.png "cartoon5.png")