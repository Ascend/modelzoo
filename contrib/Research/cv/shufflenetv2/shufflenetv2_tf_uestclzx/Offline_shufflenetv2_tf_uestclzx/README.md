## 1、原始模型
模型下载来自：由训练脚本产生的模型。桶地址：obs://modelarts-zjw/Offline_shufflenetv2/ck_model

将保存的ckpt文件转化为pb文件之后，使用ATC工具转化成om模型

## 2、转om模型
obs链接：obs://modelarts-zjw/Offline_shufflenetv2/om_model

ATC转换命令：

/home/HwHiAiUser/Ascend/ascend-toolkit/20.1.rc1/atc/bin/atc --input_shape="input:1,224,224,3" --check_report=/home/HwHiAiUser/modelzoo/shufflenetv2/device/network_analysis.report --input_format=NHWC --output="/home/HwHiAiUser/modelzoo/shufflenetv2/device/shufflenetv2" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/910model/shufflenetv2.pb" 

## 3、代码及路径解释

```
shufflenetv2
└─
  ├─README.md
  ├─LICENSE  
  ├─input                   用于存放验证集.bin文件         桶地址 obs://modelarts-zjw/Offline_shufflenetv2/input
  ├─output                  用于存放推理后的预测值.bin文件
  ├─ground_truth            用于存放标签                  桶地址 obs://modelarts-zjw/Offline_shufflenetv2/ground_truth
  ├─model                   用于存放om模型                桶地址 obs://modelarts-zjw/Offline_shufflenetv2/om_model
  ├─ck_model                用于存放checkpoint模型        桶地址 obs://modelarts-zjw/Offline_shufflenetv2/ck_model
  ├─pb_model                用于存放pb模型                桶地址 obs://modelarts-zjw/Offline_shufflenetv2/pb_model
  ├─model_freeze.py         将.ckpt模型转换为.pb模型
  ├─img_preprocess.py       将RGB图像转换为bin格式
  ├─accuracy_top1.py        验证
  ├─start_inference.sh      执行推理、验证脚本文件
```


## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 5、数据处理(这里我们直接给出处理后的数据)

从 obs://modelarts-zjw/Offline_shufflenetv2/input 下载input.tar.gz 解压至input文件夹


## 6、性能、精度测试：
这个网络是使用ShufflenetV2来实现ImageNet12数据集分类，输入为224x224的RGB图像
使用的数据集为ImageNet12, 推理输入的格式：Batch_size:1，shape:1x224x224x3.

bash start_inference.sh

![输入图片说明](https://images.gitee.com/uploads/images/2021/0120/114335_932bdda0_8511959.png "屏幕截图.png")

Inference average time without first time: 3.65 ms 推理精度为0.6132。