# SSD-ResNet34 for TensorFlow

## 概述

SSD（Single Shot MultiBox Detector）属于one - stage目标检测模型，在保证精度的同时，检测速度较快，相比同时期的Yolo和Faster R-CNN，能更好兼顾检测精度和速度，可以达到实时检测的要求。SSD原始模型以VGG16为主干网络，根据输入shape的不同通常有SSD-300、SSD500两种模型。

Ascend本次提供的是以ResNet34为主干网络、输入shape为300的SSD-ResNet34模型。

参考论文：Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector, ECCV 2016

参考实现：https://github.com/mlperf/training_results_v0.6/tree/master/Google/benchmarks/ssd/implementations/tpu-v3-32-ssd

## 默认配置

训练数据集预处理(以coco2017/Train为例，仅作为用户参考示例)
图像的输入尺寸为：300 * 300

图像输入格式：TFRecord

随机裁剪图像尺寸

随机水平翻转图像

随机颜色抖动

使用ImageNet的平均值和标准差对输入图像进行归一化

测试数据集预处理(以coco2017/Val为例，仅作为用户参考示例)
图像的输入尺寸为300 * 300 （直接将图像resize）

图像输入格式：TFRecord

使用ImageNet的平均值和标准差对输入图像进行归一化

训练超参(以coco2017/Train为例，仅作为用户参考示例)
Batch size: 单卡32，8卡32*8
动量参数: 0.9
学习率更新策略: cosine，其中decay_steps为106 epoch对应的step数，alpha为0.01
基础学习率: 0.003
基础学习率对应的batch size: 32
权重衰减参数: 0.0005
总epoch数: 115.2
warm up epoch数: 0.8

## 快速上手

1、数据集准备，该模型兼容tensorflow官网上的数据集。

请用户自行准备好数据集，包含训练集和验证集两部分，支持多种数据集，如coco、pascal voc等，请根据需要下载数据集。
解压训练集和验证集，若原数据集未区分训练集和验证集，请按照业内通过方法进行数据集划分。
将图像数据集转换为TFRecord格式，并置于<path_to_tfrecord>目录中。
将标注json文件放置于<path_to_annotations>目录中。

2、准备预训练模型

请从以下4个链接中下载主干网络ResNet34的预训练模型的所有文件。
```
https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/checkpoint 
https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.data-00000-of-00001 
https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.index 
https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.meta
```
将这些文件放置于同一目录<path_to_pretrain>中。

3、修改npu_config目录下*.json配置文件，将对应IP修改成当前IP，board_id改为本机主板ID。
  
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

4、配置数据集路径和预训练模型路径。
修改脚本exec_main.sh中的相关参数以配置数据集路径和预训练模型路径。

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
## 环境配置

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

## 开始训练

1p训练
训练脚本run_npu_1p.sh，请用户直接执行该脚本。

```
./run_npu_1p.sh
```

8p训练
训练脚本run_npu_8p.sh，请用户直接执行该脚本。

```
./run_npu_8p.sh
```

## 开始测试

默认情况下训练模式为train_and_eval，即训练全部完成后进行一次eval。若用户需要单独进行测试，请修改脚本exec_main.sh中的参数，将mode参数修改为eval。
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

1P测试指令
```
./run_npu_1p.sh
```
8P测试指令
```
./run_npu_8p.sh
```

## 脚本参数
本工程支持用户通过修改脚本exec_main.sh中的参数来改变训练中的某些配置参数，具体含义如下。
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

## 训练过程

通过“快速上手”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持1，8P网络训练。
模型存储路径为：`training/DX/result_npu`，训练脚本log中包括如下信息。
```
INFO:tensorflow:global_step...3100
I0702 20:36:15.447960 281473224802320 npu_hook.py:114] global_step...3100
INFO:tensorflow:FPS: 4474.27, learning rate: 0.02377195, loss: 7.5989103
I0702 20:36:15.448001 281473020117008 npu_hook.py:114] global_step...3100
I0702 20:36:15.448100 281473204428816 ssd_model.py:363] FPS: 4474.27, learning rate: 0.02377195, loss: 7.5989103
INFO:tensorflow:global_step...3100
INFO:tensorflow:global_step...3100
INFO:tensorflow:FPS: 4474.15, learning rate: 0.02377195, loss: 7.7068114
I0702 20:36:15.448204 281473033695248 npu_hook.py:114] global_step...3100
I0702 20:36:15.448239 281473224802320 ssd_model.py:363] FPS: 4474.15, learning rate: 0.02377195, loss: 7.7068114
I0702 20:36:15.448228 281473237356560 npu_hook.py:114] global_step...3100
INFO:tensorflow:FPS: 4474.14, learning rate: 0.02377195, loss: 8.024865
I0702 20:36:15.448304 281473020117008 ssd_model.py:363] FPS: 4474.14, learning rate: 0.02377195, loss: 8.024865
INFO:tensorflow:FPS: 4473.95, learning rate: 0.02377195, loss: 7.698079
2020-07-02 20:36:15.448485: I tf_adapter/kernels/geop_npu.cc:545] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp17_0, num_inputs:0, num_outputs:1
INFO:tensorflow:FPS: 4474.56, learning rate: 0.02377195, loss: 7.2599835
2020-07-02 20:36:15.448528: I tf_adapter/kernels/geop_npu.cc:412] [GEOP] tf session direct1caaf88f4dadeea3, graph id: 51 no need to rebuild
I0702 20:36:15.448431 281473033695248 ssd_model.py:363] FPS: 4473.95, learning rate: 0.02377195, loss: 7.698079
2020-07-02 20:36:15.448551: I tf_adapter/kernels/geop_npu.cc:753] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp17_0 ,tf session: direct1caaf88f4dadeea3 ,graph id: 51
2020-07-02 20:36:15.448558: I tf_adapter/kernels/geop_npu.cc:545] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp26_0, num_inputs:0, num_outputs:4
I0702 20:36:15.448475 281473237356560 ssd_model.py:363] FPS: 4474.56, learning rate: 0.02377195, loss: 7.2599835
```

## 验证/推理过程
通过“快速上手”中的测试指令启动单卡或者多卡测试。单卡和多卡的配置与训练过程一致。当前只能针对该工程训练出的checkpoint进行推理测，且默认仅使用最新保存的ckpt进行推理。
测试结束后会打印验证集的coco ap，如下所示。
```
INFO:tensorflow:Eval results: {'COCO/AP': 0.2542533547853277, 'COCO/AP50': 0.42539336865290517, 'COCO/AP75': 0.2632370011771454, 'COCO/APs': 0.07653635006769123, 'COCO/APm': 0.26678658340067785, 'COCO/APl': 0.40830324817885305, 'COCO/ARmax1': 0.23846155466943963, 'COCO/ARmax10': 0.34534520877717656, 'COCO/ARmax100': 0.3633561971777221, 'COCO/ARs': 0.1242796860867874, 'COCO/ARm': 0.3963178946340227, 'COCO/ARl': 0.551939979679665}
```

