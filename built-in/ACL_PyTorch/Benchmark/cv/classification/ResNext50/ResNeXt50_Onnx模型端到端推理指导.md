# ResNeXt50 Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理TopN精度统计](#61-离线推理TopN精度统计)
	-   [6.2 开源TopN精度](#62-开源TopN精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ResNeXt50论文](https://arxiv.org/abs/1611.05431)  
本文提出了一个简单的，高度模型化的针对图像分类问题的网络结构。本文的网络是通过重复堆叠building block组成的，这些building block整合了一系列具有相同拓扑结构的变体(transformations)。本文提出的简单的设计思路可以生成一种同质的，多分支的结构。这种方法产生了一个新的维度，作者将其称为基(变体的数量，the size of the set of transformations)。在ImageNet-1K数据集上，作者可以在保证模型复杂度的限制条件下，通过提升基的大小来提高模型的准确率。更重要的是，相比于更深和更宽的网络，提升基的大小更加有效。作者将本文的模型命名为ResNeXt，本模型在ILSVRC2016上取得了第二名。本文还在ImageNet-5K和COCO数据集上进行了实验，结果均表明ResNeXt的性能比ResNet好。

### 1.2 代码地址
[ResNeXt50代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.6.0
torchvision == 0.7.0
onnx == 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载pth权重文件  
[ResNeXt50预训练pth权重文件](https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth)  
文件md5sum: 1d6611049e6ef03f1d6afa11f6f9023e  
2.编写pth2onnx脚本resnext50_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3 resnext50_pth2onnx.py resnext50_32x4d-7cdf4587.pth resnext50.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./resnext50.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnext50_bs16 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在datasets/ImageNet/val_union与datasets/ImageNet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3 imagenet_torch_preprocess.py datasets/ImageNet/val_union ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3 get_info.py bin ./prep_dataset ./resnext50_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark -model_type=vision -device_id=0 -batch_size=16 -om_path=resnext50_bs16.om -input_text_path=./resnext50_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3 vision_metric_ImageNet.py result/dumpOutput_device0/ dataset/ImageNet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.62%"}, {"key": "Top2 accuracy", "value": "87.42%"}, {"key": "Top3 accuracy", "value": "90.79%"}, {"key": "Top4 accuracy", "value": "92.56%"}, {"key": "Top5 accuracy", "value": "93.69%"}]
```
### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model               Acc@1     Acc@5
ResNeXt-50-32x4d    77.618    93.698
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务
```
./benchmark -round=50 -om_path=resnext50_bs1.om -device_id=0 -batch_size=1
```
执行50次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 49 finished cost 2.635ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_resnext50_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 374.313samples/s, ave_latency: 2.67914ms
```
batch16的性能：
```
./benchmark -round=50 -om_path=resnext50_bs16.om -device_id=0 -batch_size=16
```
```
[INFO] Dataset number: 49 finished cost 30.514ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_resnext50_bs16_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 524.094samples/s, ave_latency: 1.9101ms
```
### 7.2 T4性能数据
batch1性能：
在T4机器上安装开源TensorRT
```
cd /usr/local/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/bin/
./trtexec --onnx=resnext50.onnx --fp16 --shapes=image:1x3x224x224 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch
```
[03/24/2021-03:54:47] [I] GPU Compute
[03/24/2021-03:54:47] [I] min: 1.26575 ms
[03/24/2021-03:54:47] [I] max: 4.41528 ms
[03/24/2021-03:54:47] [I] mean: 1.31054 ms
[03/24/2021-03:54:47] [I] median: 1.30151 ms
[03/24/2021-03:54:47] [I] percentile: 1.40723 ms at 99%
[03/24/2021-03:54:47] [I] total compute time: 2.9972 s
```
batch16性能：
```
./trtexec --onnx=resnext50.onnx --fp16 --shapes=image:16x3x224x224 --threads
```
```
[03/24/2021-03:57:22] [I] GPU Compute
[03/24/2021-03:57:22] [I] min: 12.5645 ms
[03/24/2021-03:57:22] [I] max: 14.8437 ms
[03/24/2021-03:57:22] [I] mean: 12.9561 ms
[03/24/2021-03:57:22] [I] median: 12.8541 ms
[03/24/2021-03:57:22] [I] percentile: 14.8377 ms at 99%
[03/24/2021-03:57:22] [I] total compute time: 3.03173 s
```
### 7.3 性能对比
batch1：2.67914/4 < 1.31054/1  
batch16：1.9101/4 < 12.9561/16  
npu的吞吐率乘4比T4的吞吐率大，即npu的时延除4比T4的时延除以batch小，故npu性能高于T4性能，性能达标。  
对于batch1与batch16，npu性能均高于T4性能1.2倍，该模型放在benchmark/cv/classification目录下。  


