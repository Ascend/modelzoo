# DenseNet121 For MindSpore
This repository provides a script and recipe to train and infer on DenseNet121 to achieve state of the art accuracy, and is tested and maintained by HuaWei.

## Table Of Contents
* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default Configuration](#default-configuration)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
    * [Training performance results](#training-performance-results)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

## Model overview

DenseNet121 is a convolution based neural network for the task of image classification. The paper describing the model can be found [here](https://arxiv.org/abs/1608.06993). HuaWei’s DenseNet121 is a implementation on [MindSpore](https://www.mindspore.cn/).

The repository also contains scripts to launch training and inference routines.

### Model architecture

DenseNet121 builds on 4 densely connected block. In every dense block, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Concatenation is used. Each layer is receiving a “collective knowledge” from all preceding layers.

 
### Default Configuration
The default configuration of this model can be found at `train.py`. The default hyper-parameters are as follows:
  - General:
    - Base Learning Rate set to 0.1 and use cosine lr
    - Global batch size set to 256 images (8p, both Training & Test)
    - Total epochs set to 120
    - Weight decay set to 0.0001, momentum set to 0.9

  - Training Dataset preprocess:
    - Input size of images is 224\*224
    - Range (min, max) of respective size of the original size to be cropped is (0.08, 1.0)
    - Range (min, max) of aspect ratio to be cropped is (0.75, 1.333)
    - Probability of the image being flipped set to 0.5
    - Randomly adjust the brightness, contrast, saturation (0.4, 0.4, 0.4)
    - Normalize the input image with respect to mean and standard deviation

  - Test Dataset preprocess:
    - Input size of images is 224\*224 (Resize to 256\*256 then crops images at the center)
    - Normalize the input image with respect to mean and standard deviation




## Setup
The following sections list the requirements in order to start training the DenseNet121 model.
### Requirements
Ensure you have the following components:
  - [MindSpore](https://www.mindspore.cn/)
  - Hardware environment with the Ascend AI processor


  For more information about how to get started with MindSpore, see the
  following sections:
  - [MindSpore's Tutorial](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
  - [MindSpore's Api](https://www.mindspore.cn/api/zh-CN/master/index.html)

## Quick Start Guide
To train your model using mixed precision with tensor cores or using FP32, perform the following steps using the default parameters of the Mask R-CNN model on the COCO 2014 dataset.

### 1. Clone the repository.
```
git clone xxx.git
cd modelzoo_densenet121/
```

### 2. Download and preprocess the dataset.
1. Download the classification dataset
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.


### 3. Start training.
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
cd scripts
sh train_8p.sh
```
The log and models will be generated in scripts/device0/output/202x-xx-xx_time_xx_xx_xx/
### 4. Start test.
for example:

```shell
cd scripts
sh test.sh
```
The result will be generated in scripts/device0/output/202x-xx-xx_time_xx_xx_xx/202x_xxxx.log


## Advanced
The following sections provide greater details of running training and inference, and the training results.

### Scripts and sample code


Descriptions of the key scripts and folders are provided below.

  - train.py - End to end to script to load data, build and train the model.
  - test.py - End to end script to load data, checkpoint and perform inference and compute mAP score. 
  - scripts/ - Contains shell scripts to train the model and perform inferences.
	-   train_8p.sh - Launches model training using 8p
	-   test.sh  - Performs inference and compute accuracy of classification.    




### Parameters

You can modify the training behaviour through the various flags in the `train.py` script. Flags in the `train.py` script are as follows:

```
  --data_dir              train data dir
  --num_classes           num of classes in dataset（default:1000)
  --image_size            image size of the dataset
  --per_batch_size        mini-batch size (default: 256) per gpu
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


### Results
Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|    120     | 75.13%/92.57% |

#### Training performance results

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |   760 img/s   |
|    8     |   6000 img/s   |


## Release notes

### Changelog

### Known Issues


