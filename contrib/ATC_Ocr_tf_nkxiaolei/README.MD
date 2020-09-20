1、原始模型https://github.com/YCG09/chinese_ocr, 下载对应的ckpt，转成pb模型
当前只是转了一部分模型，还有OCR和TXT模型待转换。

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/OCR/Ocr_Angle-model.om 

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/Ocr_Angle-model.om --output output/ --loop 100

[INFO] output data success
Inference average time: 12.737030 ms
Inference average time without first time: 12.734828 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1Batch平均推理性能12.73ms

5、精度测试：暂未开始
