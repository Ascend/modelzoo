1、原始模型https://github.com/ShawnyXiao/TextClassification-Keras, 训练10个Epoch，得到h5模型，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/FastText/fast_text_frozen.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/fast_text_frozen.om --output output/ --loop 100

[INFO] output data success
Inference average time: 0.439450 ms
Inference average time without first time: 0.436374 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x400，不带AIPP，平均推理性能0.436ms

5、精度测试：暂未开始