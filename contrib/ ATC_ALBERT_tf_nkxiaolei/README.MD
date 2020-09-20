1、原始模型https://github.com/brightmart/albert_zh, 下载albert_base_zh对应的ckpt，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/Albert/albert.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试：
./msame --model model/albert.om --output output/ --loop 100

Inference average time: 12.738640 ms \n
Inference average time without first time: 12.736172 ms \n
[INFO] unload model success, model Id is 1 \n
[INFO] Execute sample success. \n

1Batch，长度128，平均推理性能12.736ms

5、精度测试：暂未开始