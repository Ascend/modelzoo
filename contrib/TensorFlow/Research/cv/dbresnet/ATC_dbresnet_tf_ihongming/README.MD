## 1、模型功能
DBNet是一个基于分割的场景文本检测模型，能够对弯曲文本具有识别能力。

## 2、原始模型
参考https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/dbresnet/dbresnet_tf_ihongming

训练生成ckpt模型，并使用postprocess/ckpt2pb.py 转pb模型

## 3、转om模型
obs链接：obs://dbresnet50/db_resnet.om

百度网盘链接：https://pan.baidu.com/s/1lWKqM53EBf-SXeSd8R8ZNQ [提取码：rb57] 

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.1.rc1/atc/bin/atc --input_shape="input_images:1,800,800,3" --check_report=/home/HwHiAiUser/modelzoo/db_resnet/device/network_analysis.report --input_format=NHWC --output="/home/HwHiAiUser/modelzoo/db_resnet/device/db_resnet" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/db_resnet.pb" 
```

## 4、将下载的om文件放在model文件夹

## 5、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 6、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/db_resnet.om --output output/ --loop 100
```
Inference average time: 33.597620 ms
Inference average time without first time: 33.595283 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
平均推理性能33.597620ms

## 7、精度测试：

### 7.1 下载Totaltext数据集
下载totaltext数据集，解压放至./datasets中

obs链接：obs://dbresnet50/totaltext.zip

百度网盘 :https://pan.baidu.com/s/1lcbr7kPSr_zzrKkEUBrMXQ [提取码：acwh]


### 7.2 执行推理和精度计算的python脚本 
python3.7 evalution.py 2>&1 | tee eval.log

主要参数设置
```
imgpath 测试数据集地址
show_res 是否对推理结果进行可视化 True/False
```
在300张val数据集上获得测试精度为 {'precision': 0.8749, 'recall': 0.7523, 'hmean': 0.8090}

## 8、推理效果：
原图：
![原图]( ./show/img1.jpg "img1.jpg")
输出图：
![输出图]( ./show/img1_show.jpg "img1_show.jpg")
