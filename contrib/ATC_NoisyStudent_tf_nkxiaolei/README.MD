1. 转换模型，NoisyStudent的pb模型，下载地址来自于Google的modelzoo

2、下载验证数据集：
http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

3、将JPG文件转换成二进制bin文件，resize大小为800*800

4、使用msame工具进行推理

5、根据输出的bin文件计算精度。