1、原始模型https://github.com/xiaochus/MobileNetV3, 下载对应的ckpt，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/MobileNetV3-large/mobilenet-v3-large.om

3、将下载的om文件放在model文件夹

4、参考https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

5、使用nsame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/mobilenet-v3-large.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 3.100290 ms
Inference average time without first time: 3.096515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```

1Batch，shape:1x224x224x3，不带AIPP，平均推理性能3.09ms

6、精度测试：

6.1 下载ImageNet 2012 val数据集：
http://www.image-net.org/download-images

6.2 下载好的数据集JPEG图片放在任意其他目录，执行预处理脚本生成bin文件：
python3.7.5 img_preprocess.py ./originnal_pic/
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

6.3 执行推理和精度计算的shell脚本：
./start_inference.sh

50000张Val数据集在NPU上推理的top1精度为：0.755， 论文参考精度值：0.752