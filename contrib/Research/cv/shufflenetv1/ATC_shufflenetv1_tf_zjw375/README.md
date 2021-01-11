## 1、原始模型
模型下载来自：由训练脚本产生的模型。桶地址：obs://modelarts-zjw/trained_model
将保存的ckpt文件转化为pb文件之后，使用ATC工具转化成om模型

## 2、转om模型
obs链接：

ATC转换命令：
```

``` 

## 3、文件

```
shufflenetv2
└─
  ├─README.md
  ├─LICENSE  
  ├─input 用于存放验证集.bin文件
  ├─model 用于存放om模型
  ├─output 用于存放推理后的预测值.bin文件
  ├─label_output 用于存放标签.bin文件
  ├─img_preprocess.py 将RGB图像转换为bin格式
  ├─label_preprocess.py 将label保存成bin格式
  ├─accuracy_top1.py.py 验证
  ├─start_inference.sh 推理、验证脚本文件
```


## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/shufflenetv1.om --output output/ --loop 100
```

```
Batch_size:96，shape:96x224x224x3.

## 6、精度测试：
这个网络是使用ShufflenetV1来实现ImageNet12数据集分类，输入为224x224的RGB图像
使用的数据集为ImageNet12, 数据集的下载和预处理


bash start_inference.sh
```
Totol pic num: 49920, Top1 accuarcy: 0.5
```
