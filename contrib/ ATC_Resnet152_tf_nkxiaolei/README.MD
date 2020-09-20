1、参考https://github.com/statech/resnet
使用Keras创建Resnet152模型，然后将Keras模型转成tf的frozen pb

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/Resnet_152/resnet152_tf.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/resnet152_tf.om --output output/ --loop 100

[INFO] output data success
Inference average time: 11.337590 ms
Inference average time without first time: 11.331727 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x224x224x3，不带AIPP，平均推理性能11.33ms

5、精度测试：暂未开始