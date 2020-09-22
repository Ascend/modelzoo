1、原始模型https://www.huaweicloud.com/ascend/resources/modelzoo/Models/9e5c32db7fa545d2813c63586feffc32, 下载pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/InceptionV3/InceptionV3_tf.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/InceptionV3_tf.om --output output/ --loop 100

[INFO] output data success
Inference average time: 4.246270 ms
Inference average time without first time: 4.241717 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x299x229x3，不带AIPP，平均推理性能4.24ms

5、精度测试：暂未开始