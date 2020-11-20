## 1、原始模型
用户如果想自行转换om文件，可下载原生caffemodel模型和权重文件，若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/sphereface/ATC%20sphereface%20%28FP32%29%20from%20Caffe%20-%20Ascend%20200DK.zip

下载后，使用ATC模型转换工具或者执行转换命令(开启AIPP)进行转换：
```
atc --model=./sphereface.prototxt --weight=./sphereface.caffemodel --framework=0 --output=./sphereface_framework_caffe_aipp_1_batch_8_input_fp32_output_FP32 --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/sphereface/sphereface_framework_caffe_aipp_1_batch_8_input_fp32_output_FP32.om



## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/sphereface_framework_caffe_aipp_1_batch_8_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```
```
[INFO] output data success
Inference average time: 7.687250 ms
Inference average time without first time: 7.685848 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batch: 8, shape: 112 * 96 * 3，带AIPP，平均推理性能7.69ms

## 6、精度测试
待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/155901_c524d075_8113712.png "2.png")

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/155916_32ff2587_8113712.png "1.PNG")

