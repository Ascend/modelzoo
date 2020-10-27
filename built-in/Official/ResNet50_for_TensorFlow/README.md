# ResNet50 for Tensorflow

This repository provides a script and recipe to train the ResNet50 model to achieve state-of-the-art accuracy.

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

ResNet50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

This is ResNet50 V1.5 model. The difference between V1 and V1.5 is that, ResNet V1.5 set stride=2 at the 3x3 convolution layer for a bottleneck block where V1 set stride=2 at the first 1x1 convolution layer.

### Model architecture

ResNet50 builds on 4 residuals bottleneck block.

### Default configuration

The following sections introduce the default configurations and hyperparameters for ResNet50 model.

#### Optimizer

This model uses Momentum optimizer from tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.8
- LR schedule: cosine_annealing
- Batch size : 256
- Weight decay :  0.0001. We do not apply weight decay on all bias and Batch Norm trainable parameters (beta/gamma)
- Label smoothing = 0.1
- We train for:
  - 90 epochs -> configuration that reaches 76.9% top1 accuracy
  - 120 epochs -> 120 epochs is a standard for ResNet50

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - Resize to (224, 224)
  - Normalize, mean=(121, 115, 100), std=(70, 68, 71)
- For inference:
  - Resize to (224, 224)
  - CenterCrop, ratio=0.8
  - Normalize, mean=(121, 115, 100), std=(70, 68, 71)

## Setup
The following section lists the requirements to start training the ResNet50 model.
### Requirements
Ensure you have the following components:
  - Tensorflow
  - Hardware environment with the Ascend AI processor


## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd Modelzoo_Resnet50_HC
```

### 2. Download and preprocess the dataset

1. Download the classification dataset, like ImageNet2012, CIFAR10, Flower and so on.
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### 3. Train

1P training:

Set parameters in train_1p.sh.

As default:
```shell
python3.7 ../src/mains/res50.py \
        --config_file=res50_256bs_1p \
        --max_train_steps=1000 \
        --iterations_per_loop=100 \
        --debug=True \
        --eval=False \
        --model_dir=${currentDir}/d_solution/ckpt${DEVICE_ID} > ${currentDir}/log/train_${device_id}.log 2>&1
```
More parameters are in --config_file under src/configs.
```shell
cd /path/to/Modelzoo_Resnet50_HC/scripts
sh ./train_1p.sh
```

8P training is similar to the former.


### 4. Test

Evaluate after training:
1. Set --eval=True.
2. Set --config_file=resnet50_256bs_8p_eval

Evaluate while training:
1. Set --eval=True.
2. Set --config_file=resnet50_256bs_8p_eval.
3. Set `mode` in --config_file as `train_and_evaluate`.

Only Evaluate:
1. Set --eval=True.
2. Set --config_file=resnet50_256bs_8p_eval.
3. Set `mode` in --config_file as `evaluate`.

## Advanced
### Command-line options
```
  --config_file           config file name
  --max_train_steps       max train steps
  --iterations_per_loop   interations per loop
  --debug=True            debug mode
  --eval=False            if evaluate after train
  --model_dir             directory of train model
```

### Config file options

```
  --mode                  mode of train, evaluate or train_and_evaluate
  --epochs_between_evals  epoch num between evaluates while training
  --data_url              train data dir
  --num_classes           num of classes in dataset（default:1000)
  --height                image height of the dataset
  --width                 image width of the dataset
  --batch_size            mini-batch size (default: 256) per gpu
  --lr_decay_mode         type of LR schedule: exponential, cosine_annealing
  --learning_rate_maximum initial learning rate
  --num_epochs            poch num to train the model
  --warmup_epochs         warmup epoch(when batchsize is large)
  --weight_decay          weight decay (default: 1e-4)
  --momentum              momentum(default: 0.9)
  --label_smooth          whether to use label smooth in CE
  --label_smooth_factor   smooth strength of original one-hot
  --log_dir               path to save log
```

### Training process

All the results of the training will be stored in the directory specified with `--model_dir` argument.
Script will store:
 - d_solution.
 - log.
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Top1/Top5   |
| :--------: | :-----------: |
|     90     | 76.90%/93.63% |
|    120     | 77.04%/93.69% |

#### Training performance results

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |   2200 img/s   |
|    8     |   17400 img/s   |











