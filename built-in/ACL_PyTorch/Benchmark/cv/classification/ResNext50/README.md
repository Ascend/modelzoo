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
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
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

### 1.2 代码地址
[ResNeXt50代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master  
commit id:b68adcf9a9280aef02fc08daed170d74d0892361  
$\color{red}{说明：删除线用于说明READ.md必要的包含项，以下带有删除线的说明在README.md中需要删除}$  
~~优先使用本任务提供的开源代码仓，填写分支与commit id，需要从github的commits中找到commit id，commit id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交~~  


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
python3.7.5
CANN 5.0.1

pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```
~~目前推理310服务器安装的是蓝区商用版本CANN 5.0.1，库若无特殊版本要求以上三个库固定这么写，需要使用python3.7命令执行脚本，pip3.7命令安装库，torch使用1.5.0版本，如果开源模型代码导出onnx要求torch版本大于1.5.0，则使用1.8.0版本，并在此处说明~~

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
```
~~requirements.txt中需要写明本模型离线推理所有必要依赖库的具体版本，版本号即是推理310服务器上推理时使用库的版本号~~  

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
```
wget https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
```
~~优先使用训练提供的权重文件，如果训练的权重文件网上能获则需给出网址，否则需要给出从哪获取权重文件。如果训练没有提供权重则使用开源代码仓的权重文件。需要给出权重文件名与通过md5sum命令计算的权重文件md5sum值~~  
2.resnext50模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决
```
git clone https://github.com/pytorch/vision
cd vision
git reset b68adcf9a9280aef02fc08daed170d74d0892361 --hard
python3.7 setup.py install
cd ..
```
~~如果需要对模型的开源代码做修改，以打patch的形式修改后再安装：patch -p1 < ../{patch_name}.diff~~  
3.编写pth2onnx脚本resnext50_pth2onnx.py  
~~如果模型开源代码仓没有安装脚本，可以通过sys.path.append(r"./vision")添加搜索路径，然后就可以引用模型开源代码仓的函数或类~~  
 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 resnext50_pth2onnx.py resnext50_32x4d-7cdf4587.pth resnext50.onnx
```

 **模型转换要点：**  
~~对于CANN包算子有问题导致模型转换失败或需要规避才能转换成功，则需要在模型转换要点里写明定位主要过程，原因与措施~~  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明  

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./resnext50.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnext50_bs16 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py resnet /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnext50_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=resnext50_bs16.om -input_text_path=./resnext50_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.62%"}, {"key": "Top2 accuracy", "value": "87.42%"}, {"key": "Top3 accuracy", "value": "90.79%"}, {"key": "Top4 accuracy", "value": "92.56%"}, {"key": "Top5 accuracy", "value": "93.69%"}]
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上  
~~因为batch可能影响精度，如果模型支持多batch的话，精度测试需要且仅测试bs1与bs16的精度~~  

### 6.2 开源精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model               Acc@1     Acc@5
ResNeXt-50-32x4d    77.618    93.698
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
~~对于CANN包算子有问题导致精度不达标或需要规避才能达标，则需要在精度调试里写明定位主要过程，原因与措施~~  
>没有遇到精度不达标的问题，故不需要进行精度调试  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

~~性能数据需要测bs1，16，4，8，32的性能数据，且需要计算出单卡吞吐率。对于npu，bs1,16要在整个数据集上推理测性能，为了避免长期占用device，bs4,8,32也可以使用纯推理测性能~~  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 243.034, latency: 205733
[data read] throughputRate: 258.963, moduleLatency: 3.86155
[preprocess] throughputRate: 258.404, moduleLatency: 3.86991
[infer] throughputRate: 244.435, Interface throughputRate: 382.328, moduleLatency: 3.35758
[post] throughputRate: 244.435, moduleLatency: 4.09107
```
Interface throughputRate: 382.328，382.328x4=1529.312既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 173.173, latency: 288729
[data read] throughputRate: 174.62, moduleLatency: 5.72673
[preprocess] throughputRate: 174.357, moduleLatency: 5.73535
[infer] throughputRate: 173.844, Interface throughputRate: 519.634, moduleLatency: 3.36724
[post] throughputRate: 10.865, moduleLatency: 92.0383
```
Interface throughputRate: 519.634，519.634x4=2078.536既是batch16 310单卡吞吐率  
batch4性能：
```
[e2e] throughputRate: 232.98, latency: 214611
[data read] throughputRate: 235.537, moduleLatency: 4.24562
[preprocess] throughputRate: 235.147, moduleLatency: 4.25266
[infer] throughputRate: 234.437, Interface throughputRate: 492.99, moduleLatency: 3.48397
[post] throughputRate: 58.6087, moduleLatency: 17.0623
```
batch4 310单卡吞吐率：492.99x4=1971.96fps  
batch8性能：
```
[e2e] throughputRate: 211.307, latency: 236622
[data read] throughputRate: 212.246, moduleLatency: 4.71152
[preprocess] throughputRate: 211.931, moduleLatency: 4.71851
[infer] throughputRate: 211.927, Interface throughputRate: 496.378, moduleLatency: 3.45797
[post] throughputRate: 26.4906, moduleLatency: 37.7493
```
batch8 310单卡吞吐率：496.378x4=1985.512fps  
batch32性能：
```
[e2e] throughputRate: 122.942, latency: 406696
[data read] throughputRate: 123.244, moduleLatency: 8.11402
[preprocess] throughputRate: 123.143, moduleLatency: 8.12064
[infer] throughputRate: 123.207, Interface throughputRate: 377.787, moduleLatency: 4.10655
[post] throughputRate: 3.8514, moduleLatency: 259.646
```
batch32 310单卡吞吐率：377.787x4=1511.148fps  

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
~~目前T4服务器安装的cuda,cudnn,TensorRT版本如上~~  
batch1性能：
```
trtexec --onnx=resnext50.onnx --fp16 --shapes=image:1x3x224x224 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准  
```
[03/24/2021-03:54:47] [I] GPU Compute
[03/24/2021-03:54:47] [I] min: 1.26575 ms
[03/24/2021-03:54:47] [I] max: 4.41528 ms
[03/24/2021-03:54:47] [I] mean: 1.31054 ms
[03/24/2021-03:54:47] [I] median: 1.30151 ms
[03/24/2021-03:54:47] [I] percentile: 1.40723 ms at 99%
[03/24/2021-03:54:47] [I] total compute time: 2.9972 s
```
batch1 t4单卡吞吐率：1000/(1.31054/1)=763.044fps  

batch16性能：
```
trtexec --onnx=resnext50.onnx --fp16 --shapes=image:16x3x224x224 --threads
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
batch16 t4单卡吞吐率：1000/(12.9561/16)=1234.940fps  

batch4性能：
```
[05/27/2021-03:16:26] [I] GPU Compute
[05/27/2021-03:16:26] [I] min: 3.77515 ms
[05/27/2021-03:16:26] [I] max: 4.07959 ms
[05/27/2021-03:16:26] [I] mean: 3.92862 ms
[05/27/2021-03:16:26] [I] median: 3.9552 ms
[05/27/2021-03:16:26] [I] percentile: 4.07324 ms at 99%
[05/27/2021-03:16:26] [I] total compute time: 3.0054 s
```
batch4 t4单卡吞吐率：1000/(3.92862/4)=1018.169fps  

batch8性能：
```
[05/27/2021-03:14:52] [I] GPU Compute
[05/27/2021-03:14:52] [I] min: 6.52148 ms
[05/27/2021-03:14:52] [I] max: 7.22937 ms
[05/27/2021-03:14:52] [I] mean: 6.80709 ms
[05/27/2021-03:14:52] [I] median: 6.78735 ms
[05/27/2021-03:14:52] [I] percentile: 7.08972 ms at 99%
[05/27/2021-03:14:52] [I] total compute time: 3.01554 s
```
batch8 t4单卡吞吐率：1000/(6.80709/8)=1175.245fps  

batch32性能：
```
[05/27/2021-03:13:11] [I] GPU Compute
[05/27/2021-03:13:11] [I] min: 23.126 ms
[05/27/2021-03:13:11] [I] max: 26.0043 ms
[05/27/2021-03:13:11] [I] mean: 24.2826 ms
[05/27/2021-03:13:11] [I] median: 24.2343 ms
[05/27/2021-03:13:11] [I] percentile: 25.6355 ms at 99%
[05/27/2021-03:13:11] [I] total compute time: 3.05961 s
```
batch32 t4单卡吞吐率：1000/(24.2826/32)=1317.816fps  

### 7.3 性能对比
batch1：382.328x4 > 1000x1/(1.31054/1)  
batch16：519.634x4 > 1000x1/(12.9561/16)  
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。  
对于batch1与batch16，310性能均高于T4性能1.2倍，该模型放在Benchmark/cv/classification目录下。  
~~对比bs1和16，小于1倍放于Research，1-1.2倍放于Official，大于1.2倍放于Benchmark，而实际提交代码时目前都放在Research目录下~~  
 **性能优化：**  
~~对于CANN包算子有问题导致性能不达标或需要规避才能达标，则需要在性能优化里写明定位主要过程，原因与措施~~  
>没有遇到性能不达标的问题，故不需要进行性能优化  

~~如果蓝区商用版本测精度或性能不达标，蓝区最新社区CANN版本测可以达标，这里需要写出原因与最新蓝区社区CANN包版本，用最新版本测。如果是无法规避的算子缺陷导致性能不达标，这里需要添加性能不达标的原因与解决方案。如果onnx因包含自定义算子不支持推理，需要说明性能是在t4上测的在线推理，如果模型不支持batch 16，也需要说明一下~~


