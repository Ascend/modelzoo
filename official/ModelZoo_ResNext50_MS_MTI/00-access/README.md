# ResNext50 for MindSpore

This repository provides a script and recipe to train the ResNext50 model to achieve state-of-the-art accuracy.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default configuration](#default-configuration)
* [Data augmentation](#data-augmentation)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Command line arguments](#command-line-arguments)
  * [Training process](#training-process)
* [Performance](#performance)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
    * [Training performance results](#training-performance-results)


## Model overview

ResNext50 model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`

We provide resnext50_32x4d version.
	
### Model architecture


### Default configuration

The following sections introduce the default configurations and hyperparameters for ResNext50 model.

#### Optimizer

This model uses Momentum optimizer from mindspore with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.4
- LR schedule: cosine_annealing
- Batch size : 128
- Weight decay :  0.0001. We do not apply weight decay on all bias and Batch Norm trainable parameters (beta/gamma)
- Label smoothing = 0.1
- We train for:
  - 150 epochs

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
The following section lists the requirements to start training the ResNext50 model.
### Requirements

mindspore

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd modelzoo_resnext50
```

### 2. Download and preprocess the dataset

1. download dataset
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
# remenber to add $PROJECT_ROOT to PYTHON_PATH in the $ENV_SH
python $PROJECT_ROOT/launch.py --nproc_per_node=$SINGLE_NODE_WORLD_SIZE --visible_devices=$VISIBLE_DEVICES --env_sh=$ENV_SH --server_id=$SERVER_ID $RUNNING_SCRIPT $SCRIPT_ARGS
```

for example:

```shell
mkdir run_test
cd run_test
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --env_sh=/path/to/env_sh.sh --server_id=xx.xxx.xxx.xxx /path/to/train.py --per_batch_size=128 --data_dir=/path/to/dataset/train/ --is_distributed=1 --lr_scheduler=cosine_annealing --per_batch_size=128 --lr=0.4 --T_max=150 --max_epoch=150 --label_smooth=1
```

### 4. Test

```shell
mkdir run_test
cd run_test
python /path/to/launch.py --nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 --mode=test --server_id=xx.xxx.xxx.xxx --env_sh=/path/to/env_sh.sh /path/to/test.py --data_dir=/path/to/dataset/val --per_batch_size=32 --pretrained=/path/to/ckpt
```
## Advanced
### Commmand-line options

```
  --data_dir              train data dir
  --num_classes           num of classes in dataset（default:1000)
  --image_size            image size of the dataset
  --per_batch_size        mini-batch size (default: 256) per gpu
  --backbone              model architecture: resnext50
  --pretrained            path of pretrained model
  --lr_scheduler          type of LR schedule: exponential, cosine_annealing
  --lr                    initial learning rate
  --lr_epochs             epoch milestone of lr changing
  --lr_gamma              decrease lr by a factor of exponential lr_scheduler
  --eta_min               eta_min in cosine_annealing scheduler
  --T_max                 T_max in cosine_annealing scheduler
  --max_epoch             max epoch num to train the model
  --warmup_epochs         warmup epoch(when batchsize is large)
  --weight_decay          weight decay (default: 1e-4)
  --momentum              momentum(default: 0.9)
  --label_smooth          whether to use label smooth in CE
  --label_smooth_factor   smooth strength of original one-hot
  --log_interval          logging interval(dafault:100)
  --ckpt_path             path to save checkpoint
  --ckpt_interval         the interval to save checkpoint
  --is_save_on_master     save checkpoint on master or all rank
  --is_distributed        if multi device(default: 1)
  --rank                  local rank of distributed(default: 0)
  --group_size            world size of distributed(default: 1)
```

### Training process

All the results of the training will be stored in the directory specified with `--ckpt_path` argument.
Script will store:
 - checkpoints.
 - log.
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|     150     |  78.23%(TOP1)/93.92%(TOP5) |

#### Training performance results

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |   7400image/sec   |
|    1     |   970image/sec   |
