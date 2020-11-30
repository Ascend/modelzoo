## 1、原始模型
用户如果想自行转换om文件，可下载原生Tensorflow模型（.pb），若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/VGG_SSD/ATC%20VGG_SSD%20%28FP32%29%20from%20caffe%20-%20Ascend%20310.zip

下载后，使用ATC模型转换工具或者执行转换命令进行转换：
```
atc --model="vgg_ssd.prototxt" --weight="vgg_ssd.caffemodel" --soc_version=Ascend310 --framework=0 --output="vgg-ssd_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32"
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

C73环境的obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/VGG_SSD/vgg-ssd_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32_C73.om

C75环境的obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/VGG_SSD/vgg-ssd_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32_C75.om 


## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/vgg-ssd_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100

```
```
[INFO] output data success
Inference average time: 24.962560 ms
Inference average time without first time: 24.958515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batch: 1, shape: 3 * 300 * 300，不带AIPP，平均推理性能 24.96ms

## 6、精度测试
待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160255_32f676b5_8113712.png "图片4.png")