1、转换模型，NoisyStudent的pb模型，下载地址来自于Google的modelzoo

2、下载验证数据集：
http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

3、将JPG文件转换成二进制bin文件，resize大小为800*800

4、使用msame工具进行推理
转om模型，obs链接：obs://nkxiaolei88/ATC Model/NoisyStudent/noisy_student_efficientnet-l2.om
将下载的om文件放在model文件夹
使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/noisy_student_efficientnet-l2.om --output output/ --loop 100
[INFO] output data success
Inference average time: 673.064750 ms
Inference average time without first time: 673.051556 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.

1batch的平均推理时间 673.05ms


5、使用bintotxt.py,解析输出的bin文件，根据输出的bin文件计算精度。
python3.7.5 output/ fp32