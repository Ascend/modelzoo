## 模型功能

该模型用于图像去雾。

## 原始模型

参考实现 ：



原始模型网络下载地址 ：


https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/SingleImageDehaze/output_graph.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/SingleImageDehaze/deploy_vel.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
 atc --model=output_graph.pb --framework=3 --input_shape="t_image_input_to_DHGAN_generator:1,512,512,3" --output=deploy_vel --soc_version=Ascend310 --input_fp16_nodes="t_image_input_to_DHGAN_generator" --output_type= FP32 "
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/deploy_vel.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 40.927510 ms
Inference average time without first time: 40.924848 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!

```

 Batch：1,shape1: 1 * 512 * 512 * 3,不带AIPP ，平均推理性能 40.924848 ms

## 精度测试!

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0204/155526_7164ff0e_5578318.png "10992_04_0.8209.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0204/155543_ad57e082_5578318.png "out_10992_04_0.8209.png")

