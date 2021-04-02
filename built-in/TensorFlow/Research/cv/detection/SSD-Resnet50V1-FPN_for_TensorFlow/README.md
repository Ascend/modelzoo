# SSD320 v1.2 For TensorFlow

## 目录
* [模型介绍](#模型介绍)
  * [模型结构](#模型结构)
  * [默认参数](#默认参数)
  * [特性说明](#特性说明)
* [安装依赖](#安装依赖)
* [快速上手](#快速上手)
* [发布说明](#发布说明)
  * [修订记录](#修订记录)
  * [已知问题](#已知问题)

## 模型介绍

The SSD320 v1.2 model is based on the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper, which describes SSD as "a method for detecting objects in images using a single deep neural network".
This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.5x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
### 模型结构

Our implementation is based on the existing [model from the TensorFlow models repository](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config).
The network was altered in order to improve accuracy and increase throughput. Changes include:
- Replacing the VGG backbone with the more popular ResNet50.
- Adding multi-scale detection to the backbone using [Feature Pyramid Networks](https://arxiv.org/pdf/1612.03144.pdf).
- Replacing the original hard negative mining loss function with [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf).
- Decreasing the input size to 320 x 320.


### 默认参数
We trained the model for 12500 steps (27 epochs) with the following setup:
- [SGDR](https://arxiv.org/pdf/1608.03983.pdf) with cosine decay learning rate
- Learning rate base = 0.16 
- Momentum = 0.9
- Warm-up learning rate = 0.0693312
- Warm-up steps = 1000
- Batch size per GPU = 32
- Number of GPUs = 8

### 特性说明

The following features are supported by this model:

| **Feature** | **Transformer-XL** |
|:------------|-------------------:|
|混合精度 | Yes |
|分布式训练 | Yes |

## 安装依赖

1、Dllogger1.0.0（https://github.com/NVIDIA/dllogger.git）

2、pycocotools 2.0（https://github.com/philferriere/cocoapi.git）

3、mpl_toolkits

## 快速上手

### 1. git仓克隆.
```
git clone https://gitlab.huawei.com/pmail_turinghava/training_shop.git
cd /home/l00334958/training_shop/03-code/ModelZoo_SSD320V1.2_TF/00-access/
```

### 2. 数据集和初始模型下载. 

数据集[COCO 2017](http://cocodataset.org/#download) 和初始模型

下载方法1：
```
download_all.sh nvidia_ssd <data_dir_path> <checkpoint_dir_path>
```

Data will be downloaded, preprocessed to tfrecords format and saved in the `<data_dir_path>` directory (on the host).
Moreover the script will download pre-trained RN50 checkpoint in the `<checkpoint_dir_path>` directory

下载方法2（10.136.181.84）：
```
 /autotest/modelzoo_datasets/CV/COCO2017/coco2017_tfrecords  下载到/data/目录
 /autotest/modelzoo_datasets/CV/COCO2017/annotations  下载到/data/目录
 /autotest/modelzoo_GPU_ckpts/CV/SSD320V1.2_TF/backbone_pretrain/ 下载到/checkpoints目录
 ```

### 3. 训练

单P训练，以数据目录为`/data`、checkpoints目录为 `/checkpoints` 为例:

```
source ./env.sh
bash npu_train_1p.sh
```

8P训练，以数据目录为`/data`、checkpoints目录为 `/checkpoints` 为例:

```
source ./env.sh
# SSD320 v1.2 For TensorFlow

## 目录
* [模型介绍](#模型介绍)
  * [模型结构](#模型结构)
  * [默认参数](#默认参数)
  * [特性说明](#特性说明)
* [安装依赖](#安装依赖)
* [快速上手](#快速上手)
* [发布说明](#发布说明)
  * [修订记录](#修订记录)
  * [已知问题](#已知问题)

## 模型介绍

The SSD320 v1.2 model is based on the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper, which describes SSD as "a method for detecting objects in images using a single deep neural network".
This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.5x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
### 模型结构

Our implementation is based on the existing [model from the TensorFlow models repository](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config).
The network was altered in order to improve accuracy and increase throughput. Changes include:
- Replacing the VGG backbone with the more popular ResNet50.
- Adding multi-scale detection to the backbone using [Feature Pyramid Networks](https://arxiv.org/pdf/1612.03144.pdf).
- Replacing the original hard negative mining loss function with [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf).
- Decreasing the input size to 320 x 320.


### 默认参数
We trained the model for 12500 steps (27 epochs) with the following setup:
- [SGDR](https://arxiv.org/pdf/1608.03983.pdf) with cosine decay learning rate
- Learning rate base = 0.16 
- Momentum = 0.9
- Warm-up learning rate = 0.0693312
- Warm-up steps = 1000
- Batch size per GPU = 32
- Number of GPUs = 8

### 特性说明

The following features are supported by this model:

| **Feature** | **Transformer-XL** |
|:------------|-------------------:|
|混合精度 | Yes |
|分布式训练 | Yes |

## 安装依赖

1、Dllogger1.0.0（https://github.com/NVIDIA/dllogger.git）

2、pycocotools 2.0（https://github.com/philferriere/cocoapi.git）

3、mpl_toolkits

## 快速上手

### 1. git仓克隆.
```
git clone https://gitlab.huawei.com/pmail_turinghava/training_shop.git
cd /home/l00334958/training_shop/03-code/ModelZoo_SSD320V1.2_TF/00-access/
```

### 2. 数据集和初始模型下载. 

数据集[COCO 2017](http://cocodataset.org/#download) 和初始模型

下载方法1：
```
download_all.sh nvidia_ssd <data_dir_path> <checkpoint_dir_path>
```

Data will be downloaded, preprocessed to tfrecords format and saved in the `<data_dir_path>` directory (on the host).
Moreover the script will download pre-trained RN50 checkpoint in the `<checkpoint_dir_path>` directory

下载方法2（10.136.181.84）：
```
 /autotest/modelzoo_datasets/CV/COCO2017/coco2017_tfrecords  下载到/data/目录
 /autotest/modelzoo_datasets/CV/COCO2017/annotations  下载到/data/目录
 /autotest/modelzoo_GPU_ckpts/CV/SSD320V1.2_TF/backbone_pretrain/ 下载到/checkpoints目录
 ```

### 3. 训练

单P训练，以数据目录为`/data`、checkpoints目录为 `/checkpoints` 为例:

```
source ./env.sh
cd test;
bash train_full_1p.sh （全量训练）
bash train_performance_1p.sh --num_train_steps=10 （性能测试）
```


### 4. 验证

基于ckpt做eval，以数据目录为`/data`、checkpoints目录为 `/checkpoints` 为例:

```
source ./env.sh
cd models/research
bash examples/SSD320_evaluate.sh /checkpoints/
```

基于pb做eval，以数据目录为`/data`、pb目录为 `./scripts/savedModel` 为例:

```
source ./env.sh
cd scripts
bash freeze_inference_graph.sh
cd models/research
bash examples/SSD320_evaluate_pb.sh ../../scripts/savedModel/
```

## 发布说明

### 修订记录

2021-03
 * 提交混合计算下训练流程OK的版本


## 已知问题
1、训练全下沉失败，DTS202102090NLYJBP1K00，已通过开启混合计算规避
2、eval阻塞，DTS202103040WIGECP1G00
3、混合计算下，训练性能太差，分析中

