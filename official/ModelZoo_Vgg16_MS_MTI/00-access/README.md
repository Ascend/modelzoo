# VGG16 for MindSpore

This repository provides a script and recipe to train the VGG16 model to achieve state-of-the-art accuracy.

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

VGG16 is a convolutional neural network architecture, its name VGG16 comes from the fact that it has 16 layers. Its layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

VGG16 model from
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`

### Model architecture


### Default configuration

The following sections introduce the default configurations and hyperparameters for VGG16 model.

#### Optimizer

This model uses Momentum optimizer from mindspore with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.01
- LR schedule: cosine_annealing
- Batch size : 32
- Weight decay :  0.0001. We do not apply weight decay on all bias and Batch Norm trainable parameters (beta/gamma)
- Label smoothing = 0.1
- We train for:
  - 150 epochs for a standard training process

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - RandomResizeCrop, scale=(0.08, 1.0), ratio=(0.75, 1.333)
  - RandomHorizontalFlip, prob=0.5
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
cd modelzoo_vgg16
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
python $PROJECT_ROOT/mindvision/common/distributed/launch.py --nproc_per_node=$SINGLE_NODE_WORLD_SIZE --visible_devices=$VISIBLE_DEVICES --env_sh=$ENV_SH --server_id=$SERVER_ID $RUNNING_SCRIPT $SCRIPT_ARGS
```

for example:

```shell
mkdir run_test
cd run_test
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --env_sh=/path/to/env_sh.sh --server_id=xx.xxx.xxx.xxx /path/to/train.py --per_batch_size=32 --data_dir=/path/to/dataset/train/ --is_distributed=1 --lr_scheduler=cosine_annealing --weight_decay=0.0001 --lr=0.01 --T_max=150 --max_epoch=150 --warmup_epochs=0 --label_smooth=1
```

### 4. Test

```shell
mkdir run_test
cd run_test
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --mode=test --server_id=xx.xxx.xxx.xxx --env_sh=/path/to/env_sh.sh /path/to/test.py --data_dir=/path/to/dataset/val --per_batch_size=32 --pretrained=/path/to/ckpt
```

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|    150     | 74.22%/92.08% |

#### Training performance results

| **GPUs** | train performance |
| :------: | :---------------: |
|    1     |   870  img/s   |
|    8     |   4100 img/s   |











