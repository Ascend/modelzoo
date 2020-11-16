## 1、原始模型
用户如果想自行转换om文件，可下载原生caffemodel模型和权重文件，若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/faster_rcnn/ATC%20faster_rcnn%28fp32%29%20from%20caffe%20-%20Ascend%20310.zip

下载后，使用ATC模型转换工具或者执行转换命令进行转换：
```
atc --model=faster_rcnn.prototxt --weight=faster_rcnn.caffemodel --framework=0 --output=vgg-ssd_framework_caffe_ascend310_aipp_0_batch_1_input_fp32_output_FP32.om --soc_version=Ascend310  --input_shape="data:1,3,-1,-1;im_info:1,3" --dynamic_image_size="600,600;800,800"
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/faster_rcnn/faster_rcnn_framework_caffe_ascend310_aipp_1_batch_1_input_fp32_output_FP32.om 


## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/faster_rcnn_framework_caffe_ascend310_aipp_1_batch_1_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
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

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/155216_6451a809_8113712.png "bicycle.png")