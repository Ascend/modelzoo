## 模型功能

该模型用于将普通照片转换为宫崎骏卡通风格的图片。

## 原始模型

参考实现 ：

https://github.com/TachibanaYoshino/AnimeGANv2 

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/cartoonization/AnimeGANv2/AnimeGANv2_Hayao.pb

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/cartoonization/AnimeGANv2/AnimeGANv2_Hayao.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=AnimeGANv2_Hayao.pb  --input_shape="generator_input:1,256,256,3"  --framework=3  --output=AnimeGANv2_Hayao  --soc_version=Ascend310 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/AnimeGANv2_Hayao.om --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 2032.205600 ms
Inference average time without first time: 2032.041667 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

Batch: 1, shape: 256 * 256 * 3，平均推理性能 2032.041667ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/182419_05ed823c_8113712.jpeg "hayao_44.jpg")

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/182428_247cf211_8113712.jpeg "hayao_AE86.jpg")
