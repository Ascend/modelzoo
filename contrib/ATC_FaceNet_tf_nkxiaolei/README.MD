1、原始模型https://github.com/davidsandberg/facenet, 下载对应的ckpt，转成pb模型
原有的模型输入节点不能被ATC工具识别，需要进行改造，输入节点改成Input

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/FaceNet/facenet_modify.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/facenet_modify.om --output output/ --loop 100

[INFO] output data success
Inference average time: 24.251630 ms
Inference average time without first time: 24.241586 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x160x160x3，不带AIPP，平均推理性能24.24ms

5、精度测试：暂未开始