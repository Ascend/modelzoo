1、原始模型https://github.com/qubvel/efficientnet, 下载对应的ckpt，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/EfficientNet-B8/efficientnet-b8.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/efficientnet-b8.om --output output/ --loop 100

[INFO] output data success
Inference average time: 139.019250 ms
Inference average time without first time: 139.011384 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x672x672x3，不带AIPP，平均推理性能139ms

5、精度测试：暂未开始