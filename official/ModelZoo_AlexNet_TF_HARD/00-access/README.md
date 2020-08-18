# Alexnet for Tensorflow 

This repository provides a script and recipe to train the AlexNet model .

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


​    

## Model overview

AlexNet model from
`Alex Krizhevsky. "One weird trick for parallelizing convolutional neural networks". <https://arxiv.org/abs/1404.5997>.`
reference implementation:  <https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html#alexnet>
### Model architecture



### Default configuration

The following sections introduce the default configurations and hyperparameters for AlexNet model.

#### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.06
- LR schedule: cosine_annealing
- Batch size : 128 
- Weight decay :  0.0001. 
- Label smoothing = 0.1
- We train for:
  - 150 epochs ->  60.1% top1 accuracy

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
The following section lists the requirements to start training the Alexnet model.
### Requirements

Tensorflow
NPU environmemnt

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd  Model_zoo_Alexnet_HARD
```

### 2. Download and preprocess the dataset

1. down load the imagenet dataset
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### 3. Train
- train on single NPU
    - **edit** *scripts/train_alexnet_1p.sh*( see example below)
    - bash scripts/run_npu_1p.sh
- train on 8 NPUs
    - **edit** *scripts/train_alexnet_8p.sh*(see example below)
    - bash scripts/run_npu_8p.sh 


for example:
- case for single NPU
    - In scripts/train_alexnet_1p.sh , python scripts part should look like as follows. For more detailed command lines arguments, please refer to [Command line arguments](#command-line-arguments)
```shell
python3.7 ${EXEC_DIR}/train.py --rank_size=1 \
	--iterations_per_loop=100 \
	--batch_size=256 \
	--data_dir=/path/to/dataset \
	--mode=train \
	--lr=0.015 \
	--log_dir=./model_1p > ./train_${device_id}.log 2>&1 
```
run the program  
```
bash scripts/run_npu_1p.sh
```
- case for 8 NPUs
    - In `scripts/train_alexnet_8p.sh` , python scripts part should look like as follows.
```shell 
python3.7 ${EXEC_DIR}/train.py --rank_size=8 \
	--iterations_per_loop=100 \
	--batch_size=128 \
	--data_dir=/path/to/dataset \
	--mode=train \
	--lr=0.06 \
	--log_dir=./model_8p > ./train_${device_id}.log 2>&1 
```
run the program  
```
bash scripts/run_npu_1p.sh
```

### 4. Test
- same procedure as training except 2 following modifications
    - change `--mode=train` to `--mode=evaluate`
    - add `--checkpoint_dir=/path/to/checkpoints`
    - comment out `rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/` in the `train_alexnet_8p.sh` or `train_alexnet_1p.sh` if you are using one single NPU for test


## Advanced
### Command-line options

```
  --data_dir                        train data dir
  --num_classes                     num of classes in ImageNet（default:1000)
  --image_size                      image size of the dataset
  --batch_size                      mini-batch size (default: 128) per npu
  --pretrained                      path of pretrained model
  --lr                              initial learning rate
  --max_epochs                      max epoch num to train the model
  --warmup_epochs                   warmup epoch(when batchsize is large)
  --weight_decay                    weight decay (default: 1e-4)
  --momentum                        momentum(default: 0.9)
  --label_smoothing                 use label smooth in CE, default 0.1
  --save_summary_steps              logging interval(dafault:100)
  --log_dir                         path to save checkpoint and log
  --log_name                        name of log file
  --save_checkpoints_steps          the interval to save checkpoint
  --mode                            mode to run the program (train, evaluate)
  --checkpoint_dir                  path to checkpoint for evaluation
  --max_train_steps                 max number of training steps 
  --synthetic                       whether to use synthetic data or not
  --version                         weight initialization for model
  --do_checkpoint                   whether to save checkpoint or not 
  --rank_size                       local rank of distributed(default: 0)
  --group_size                      world size of distributed(default: 1)
  --max_train_steps                 number of training step , default : None, when set ,it will override the max_epoch
```
for a complete list of options, please refer to `train.py`
### Training process

All the results of the training will be stored in the directory `results`.
Script will store:

 - checkpoints
 - log

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|    150     | 60.12%/82.06% |

#### Training performance results

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |   25000+  img/s   |











