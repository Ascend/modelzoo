1、模型下载来自：https://github.com/opconty/keras-shufflenetV2
将保存的h5文件转化为pb文件之后，使用ATC工具转化成om模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/ShufflenetV2/shufflenetv2_emotion_recogn.om 

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/shufflenetv2_emotion_recogn.om --output output/ --loop 100

[INFO] output data success
Inference average time: 0.888980 ms
Inference average time without first time: 0.885677 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x48x48x1，不带AIPP，平均推理性能0.886ms

5、精度测试：暂未开始