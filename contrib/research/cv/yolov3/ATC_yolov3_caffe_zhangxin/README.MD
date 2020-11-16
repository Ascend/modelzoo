## 1、原始模型
用户如果想自行转换om文件，可下载原生caffe模型及AIPP配置文件，若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/ATC%20Yolov3%20%28FP32%29%20from%20Caffe%20-%20Ascend%20310.zip 

下载后，使用ATC模型转换工具或者执行转换命令(开启AIPP)进行转换：
```
atc --model=yolov3.prototxt --weight=yolov3.caffemodel --framework=0 --output=yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om --soc_version=Ascend310 --insert_op_conf=aipp_nv12.cfg
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om 


## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../../../ATC_YOLOv3_caffe/model/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om  --output ../../../ATC_YOLOv3_caffe/msame_infer --outfmt TXT --loop 100
```
```
Inference average time: 12.098610 ms
Inference average time without first time: 12.097303 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batch: 1, shape: 416* 416 *3 ，带AIPP，平均推理性能12.10ms


## 6、精度测试
待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160411_c5731771_8113712.jpeg "yolo-caffe.jpg")

