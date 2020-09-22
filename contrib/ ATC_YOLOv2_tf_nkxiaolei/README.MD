1、从darknet官网下载yolov2的weights文件，转成pb文件，使用ATC工具转成om

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/YoloV2/yolov2.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/yolov2.om --output output/ --loop 100

[INFO] output data success
Inference average time: 32.784820 ms
Inference average time without first time: 32.780202 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x608x608x3，不带AIPP，平均推理性能32.78ms

5、精度测试：暂未开始