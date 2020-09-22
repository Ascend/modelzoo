1、模型下载地址：
https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
xception65_coco_voc_trainval

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/DeepLapV3_Plus/deeplabv3_plus.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/deeplabv3_plus.om --output output/ --loop 100

[INFO] output data success
Inference average time: 359.945320 ms
Inference average time without first time: 359.918535 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch，shape:1x513x513x3，不带AIPP，平均推理性能359.91ms

5、精度测试：暂未开始

