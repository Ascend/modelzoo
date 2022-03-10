# SSD-ResNet34 for TensorFlow

## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)

## 基本信息

**发布者（Publisher）：Huawei**
**应用领域（Application Domain）：Object Detection
**版本（Version）：1.1
**修改时间（Modified） ：2021.11.19
**大小（Size）：214KB
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：Mixed
**处理器（Processor）：昇腾910
**应用级别（Categories）：Official
**描述（Description）：基于tensorflow实现，以Resnet34为backbone的SSD目标检测网络。


## 概述

SSD（Single Shot MultiBox Detector）属于one - stage目标检测模型，在保证精度的同时，检测速度较快，相比同时期的Yolo和Faster R-CNN，能更好兼顾检测精度和速度，可以达到实时检测的要求。SSD原始模型以VGG16为主干网络，根据输入shape的不同通常有SSD-300、SSD500两种模型。

Ascend本次提供的是以ResNet34为主干网络、输入shape为300的SSD-ResNet34模型。

- 参考论文
     
     https://arxiv.org/abs/1512.02325

- 参考实现

    https://github.com/mlperf/training_results_v0.6/tree/master/Google/benchmarks/ssd/implementations/tpu-v3-32-ssd

- 适配昇腾 AI 处理器的实现：

    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/detection/SSD-Resnet34_ID0048_for_TensorFlow


### 默认配置

- 训练数据集预处理(以coco2017/Train为例，仅作为用户参考示例)

  图像的输入尺寸为：300 * 300

  图像输入格式：TFRecord

  随机裁剪图像尺寸

  随机水平翻转图像

  随机颜色抖动

  使用ImageNet的平均值和标准差对输入图像进行归一化

- 测试数据集预处理(以coco2017/Val为例，仅作为用户参考示例)

  图像的输入尺寸为300 * 300 （直接将图像resize）

  图像输入格式：TFRecord

  使用ImageNet的平均值和标准差对输入图像进行归一化

- 训练超参(以coco2017/Train为例，仅作为用户参考示例)

  - Batch size: 单卡32，8卡32*8

  - 动量参数: 0.9

  - 学习率更新策略: cosine，其中decay_steps为106 epoch对应的step数，alpha为0.01

  - 基础学习率: 0.003

  - 基础学习率对应的batch size: 32

  - 权重衰减参数: 0.0005

  - 总epoch数: 115.2

  - warm up epoch数: 0.8

## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>

## 安装依赖

- 根据requirements.txt安装必要的依赖包


## 快速上手

### 数据集准备

- 请用户自行准备好数据集，包含训练集和验证集两部分，支持多种数据集，如coco、pascal voc等，请根据需要下载数据集。

- 解压训练集和验证集，若原数据集未区分训练集和验证集，请按照业内通过方法进行数据集划分。

- 将图像数据集转换为TFRecord格式，并置于<path_to_tfrecord>目录中。

- 将标注json文件放置于<path_to_annotations>目录中。


### 预训练模型准备

请自行下载主干网络ResNet34的预训练模型的所有文件。

将这些文件放置于同一目录<path_to_pretrain>中。


### Docker容器场景

- 编译镜像
```bash
docker build -t ascend-ssd .
```

- 启动容器实例
```bash
bash docker_start.sh
```

参数说明:

```
#!/usr/bin/env bash
docker_image=$1 \   #接受第一个参数作为docker_image
data_dir=$2 \       #接受第二个参数作为训练数据集路径
model_dir=$3 \      #接受第三个参数作为模型执行路径
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #docker使用卡数，当前使用0~7卡
 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #训练数据集路径
        -v ${model_dir}:${model_dir} \  #模型执行路径
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image为镜像名称
        /bin/bash
```

执行docker_start.sh后带三个参数：
  - 生成的docker_image
  - 数据集路径
  - 模型执行路径
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```



### 检查json

修改 `npu_config` 目录下 `*.json` 配置文件，将对应IP修改成当前IP，board_id改为本机主板ID。

1P 情况下不需要rank_table配置文件，8P rank_table json配置文件如下。

```
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "",
            "instance_count": "8",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.100.101"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "1",
                            "device_ip": "192.168.101.101"
                        }
                    ],
                    "rank_id": "1",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "2",
                            "device_ip": "192.168.102.101"
                        }
                    ],
                    "rank_id": "2",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "3",
                            "device_ip": "192.168.103.101"
                        }
                    ],
                    "rank_id": "3",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "4",
                            "device_ip": "192.168.100.100"
                        }
                    ],
                    "rank_id": "4",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "5",
                            "device_ip": "192.168.101.100"
                        }
                    ],
                    "rank_id": "5",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "6",
                            "device_ip": "192.168.102.100"
                        }
                    ],
                    "rank_id": "6",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "7",
                            "device_ip": "192.168.103.100"
                        }
                    ],
                    "rank_id": "7",
                    "server_id": "0.0.0.0"
                }
            ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0",
        "eth1",
        "eth2",
        "eth3",
        "eth4",
        "eth5",
        "eth6",
        "eth7"
    ],
    "para_plane_nic_num": "8",
    "status": "completed"
}
```


### 关键配置修改

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


配置数据集路径和预训练模型路径：

- 修改脚本 `exec_main.sh` 中的相关参数以配置数据集路径和预训练模型路径

```
python3 $3/ssd_main.py  \
                     --mode=train_and_eval \
                     --train_batch_size=32 \
                     --training_file_pattern=${path_to_tfrecord}/train2017* \
                     --resnet_checkpoint=${path_to_pretrain}/model.ckpt-28152 \
                     --validation_file_pattern=${path_to_tfrecord}/val2017* \
                     --val_json_file=${path_to_annotations}/instances_val2017.json \
                     --eval_batch_size=32 \
                     --model_dir=result_npu
```


### 运行示例

#### 模型训练

- 1p训练

训练脚本`run_npu_1p.sh`，请用户直接执行该脚本。

```
./run_npu_1p.sh
```

- 8p训练

训练脚本`run_npu_8p.sh`，请用户直接执行该脚本。

```
./run_npu_8p.sh
```

#### 模型评估

默认情况下训练模式为train_and_eval，即训练全部完成后进行一次eval。若用户需要单独进行测试，请修改脚本`exec_main.sh`中的参数，将mode参数修改为eval。
```
python3 $3/ssd_main.py  \
                     --mode=eval \
                     --train_batch_size=32 \
                     --training_file_pattern=${path_to_tfrecord}/train2017* \
                     --resnet_checkpoint=${path_to_pretrain}/model.ckpt-28152 \
                     --validation_file_pattern=${path_to_tfrecord}/val2017* \
                     --val_json_file=${path_to_annotations}/instances_val2017.json \
                     --eval_batch_size=32 \
                     --model_dir=result_npu
```

- 1P测试指令

```
./run_npu_1p.sh
```

- 8P测试指令

```
./run_npu_8p.sh
```

## 高级参考

### 脚本和示例代码<a name="section08421615141513"></a>

```
├── npu_config
│   └── 8p.json
├── object_detection
│   ├── __init__.py
│   ├── argmax_matcher.py
│   ├── box_coder.py
│   ├── box_list.py
│   ├── faster_rcnn_coder.py
│   ├── matcher.py
│   ├── preprocessor.py
│   ├── region_similarity_calculator.py
│   ├── shape_utils.py
│   ├── target_assigner.py
│   └── tf_example_decoder.py
├── test
│   ├── train_full_1p.sh                
│   ├── train_performance_1p.sh
│   └── train_performance_8p.sh
├── Dockerfile
├── LICENSE
├── README.md  
├── coco_metric.py   
├── create_coco_tf_record.py  
├── dataloader.py  
├── docker_start.sh  
├── exec_main.sh 
├── frozen_graph.py 
├── infer_from_pb.py
├── modelzoo_level.txt                
├── requirements.txt   
├── run_npu_1p.sh 
├── run_npu_8p.sh 
├── ssd_1p.sh 
├── ssd_8p.sh 
├── ssd_architecture.py 
├── ssd_constants.py
├── ssd_main.py               
└── ssd_mmodel.py


```

### 脚本参数

本工程支持用户通过修改脚本`exec_main.sh`中的参数来改变训练中的某些配置参数，具体含义如下。

```
python3 $3/ssd_main.py  \
            --mode=train_and_eval \   # 模式，支持train_and_eval，train，eval三种模式
            --train_batch_size=32 \   # 训练时每个npu设备上的batch size
            --training_file_pattern=${path_to_tfrecord}/train2017* \    # 训练集tfrecord路径
            --resnet_checkpoint=${path_to_pretrain}/model.ckpt-28152 \  # 预训练模型路径
            --validation_file_pattern=${path_to_tfrecord}/val2017* \    # 验证集tfrecord路径
            --val_json_file=${path_to_annotations}/instances_val2017.json \   # 验证集标注文件
            --eval_batch_size=32 \   # 测试时每个npu设备上的batch size
            --model_dir=result_npu   # 模型保存路径
```



