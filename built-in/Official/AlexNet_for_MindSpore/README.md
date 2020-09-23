# AlexNet for MindSpore

This repository provides a script and recipe to train the AlexNet model to achieve state-of-the-art accuracy.

## Table Of Contents

- Model overview

  - Model architecture

  - Default configuration

    - Optimizer
- Data augmentation
  
- Setup

  - Requirements

- Quick Start Guide

- Performance

  - Results

    

## Model overview

AlexNet is a convolutional neural network architecture. Its layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

AlexNet model from
    `"One weird trick for parallelizing convolutional neural networks" <https://arxiv.org/abs/1404.5997>`_ paper.

### Model architecture

AlexNet consists of one 11x11 convolution kernel, one 5x5 convolution kernel and three 3x3 convolution kernels, which is fairly easy to understand.

### Default configuration

The following sections introduce the default configurations and hyperparameters for AlexNet model.

#### Optimizer

This model uses Momentum optimizer from mindspore with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.13
- LR schedule: cosine_annealing
- Batch size : 256
- Weight decay :  0.0001
- Label smoothing = 0.1
- warmup epochs : 5
- We train for:
  - 150 epochs for a standard training process using ImageNet

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - RandomResizeCrop, scale=(0.08, 1.0), ratio=(0.75, 1.333)
  - RandomHorizontalFlip, prob=0.5
  - RandomColorAdjust, brightness=0.4, contrast=0.4, saturation=0.4
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- For inference:
  - Resize to (256, 256)
  - CenterCrop to (224, 224)
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

## Setup

### Requirements

Ensure you have the following components:
  - [MindSpore](https://www.mindspore.cn/)
  - Hardware environment with the Ascend AI processor


  For more information about how to get started with MindSpore, see the
  following sections:
  - [MindSpore's Tutorial](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
  - [MindSpore's Api](https://www.mindspore.cn/api/zh-CN/master/index.html)

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd modelzoo_alexnet
```

### 2. Download and preprocess the dataset

1. down load the classification dataset
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### 3. Train

```shell
# $PROJECT_ROOT is the project path
# $SINGLE_NODE_WORLD_SIZE is the number of devices
# $VISIBLE_DEVICES is the running device
# $ENV_SH is your own 1node environment shell script
# $SERVER_ID is your device ip
# $RUNNING_SCRIPT is the running script for distributing
# $SCRIPT_ARGS is the running script args
# $SCRIPT_ARGS is the running script args
# remenber to add $PROJECT_ROOT to PYTHON_PATH in the $ENV_SH
python $PROJECT_ROOT/.../launch.py --nproc_per_node=$SINGLE_NODE_WORLD_SIZE --visible_devices=$VISIBLE_DEVICES --env_sh=$ENV_SH --server_id=$SERVER_ID $RUNNING_SCRIPT $SCRIPT_ARGS
```

for example:

```
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --env_sh=/path/to/env_sh.sh --server_id=xx.xxx.xxx.xxx /path/to/train.py --per_batch_size=256 --data_dir=/path/to/dataset/train/ --is_distributed=1 --lr_scheduler=cosine_annealing --weight_decay=0.0001 --lr=0.13 --T_max=150 --max_epoch=150 --warmup_epochs=5 --label_smooth=1 --backbone=alexnet
```

### 4. Test
The test command is as follows.(note: for testing, the current version needs to manually modify nn.Dropout(keep_prob=0.65) to nn.Dropout(keep_prob =1.0) in alexnet.py)
```
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --mode=test --server_id=xx.xxx.xxx.xxx --env_sh=/path/to/env_sh.sh /path/to/test.py --data_dir=/path/to/dataset/val --per_batch_size=32 --pretrained=/path/to/ckpt
```

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|    150     | 60.49%/82.56% |

#### Training performance results

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |     8500 img/s    |

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |    20000 img/s    |











