## 1、原始模型
用户如果想自行转换om文件，可下载原生caffemodel模型和权重文件，若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/ATC%20colorization%20%28FP32%29%20from%20caffe%20-%20Ascend%20310.zip

下载后，使用ATC模型转换工具或者执行转换命令进行转换：
```
atc --output_type=FP32 --input_shape="data_l:1,1,224,224" --weight="./colorization.caffemodel"  --input_format=NCHW --output="./colorization" --soc_version=Ascend310 --framework=0 --save_original_model=false --model="./colorization.prototxt" 
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization_framework_caffe_aipp_0_batch_1_input_fp32_output_FP32.om


## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
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

## 6、精度测试
待完善

推理效果

![图片1](C:\Users\Administrator\Desktop\图片1.png)

![图片2](C:\Users\Administrator\Desktop\图片2.png)