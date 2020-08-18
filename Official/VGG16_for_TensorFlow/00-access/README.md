# VGG16 for Tensorflow 

This repository provides a script and recipe to train the VGG16 model .

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

In this repository, we implement VGG16 from paper [Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition.](https://arxiv.org/abs/1409.1556).

VGG16 is a convolutional neural network architecture, its name VGG16 comes from the fact that it has 16 layers. This is an implementation of the official VGG16 network, written mainly in Tensorflow and can run on Ascend 910.

### Model architecture

The model architecture can be found from the reference paper.

### Default configuration

The following sections introduce the default configurations and hyperparameters for VGG16 model.

#### Optimizer

This model uses Nesterov Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.01
- LR schedule: cosine_annealing
- Batch size : 32*8 
- Weight decay :  0.0001. 
- Label smoothing = 0.1
- We train for:
  - 150 epochs for a standard training process using ImageNet2012

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
The following section lists the requirements to start training the VGG16 model.
### Requirements

Tensorflow 1.15.0

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd modelzoo_vgg16_TF
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset
2. Generate tfrecord files following [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).
3. The train and validation tfrecord files are under the path/data directories.

### 3. Train
- train on a single NPU
    - **edit** *scripts/run_1p.sh* and *scripts/train_1p.sh* (see example below)
    - bash run_1p.sh
- train on 8 NPUs
    - **edit** *scripts/run_8p.sh* and *scripts/train_8p.sh* (see example below)
    - bash run_8p.sh 

Examples:
- Case for single NPU
    - Modify the ID of NPU in *device_group* in *scripts/run_1p.sh*, default ID is *0*.
    - In *scripts/train_1p.sh* , python scripts part should look like as follows. For more detailed command lines arguments, please refer to [Command line arguments](#command-line-arguments)
        ```shell
        python3.7 ${dname}/train.py --rank_size=1 \
            --mode=train \
            --max_train_steps=100 \
            --iterations_per_loop=10 \
            --data_dir=/opt/npu/slimImagenet \
            --display_every=10 \
            --log_dir=./model_1p \
            --log_name=vgg16_1p.log
        ```
    - Run the program  
        ```
        bash run_1p.sh
        ```
- Case for 8 NPUs
    - Modify the ID of NPU in *device_group* in *scripts/run_8p.sh*, default ID is *0,1,2,3,4,5,6,7*.
    - In *scripts/train_8p.sh* , python scripts part should look like as follows.
        ```shell 
        python3.7 ${dname}/train.py --rank_size=8 \
            --mode=train_and_evaluate \
            --max_epochs=150 \
            --iterations_per_loop=5004 \
            --epochs_between_evals=5 \
            --data_dir=/opt/npu/slimImagenet \
            --lr=0.01 \
            --log_dir=./model_8p \
            --log_name=vgg16_8p.log
        ```
    - Run the program  
        ```
        bash run_8p.sh
        ```

### 4. Test
- Same procedure as training except 2 following modifications
    - change `--mode=train` to `--mode=evaluate`
    - add `--eval_dir=path/eval`
     ```shell 
    python3.7 ${dname}/train.py --rank_size=1 \
        --mode=evaluate \
        --data_dir=/opt/npu/slimImagenet \
        --eval_dir=${dname}/scripts/result/8p/0/model_8p \
        --log_dir=./ \
        --log_name=eval_vgg16.log > eval.log
    ```
    run the program  
    ```
    bash test.sh
    ```


## Advanced
### Commmand-line options

```
  --rank_size                       number of NPUs to use (default: 0)
  --mode                            mode to run the program (default: train_and_evaluate)
  --max_train_steps                 max number of training steps (default: 100)
  --iterations_per_loop             number of steps to run in devices each iteration (default: 10)
  --max_epochs                      number of epochs to train the model (default: None)
  --epochs_between_evals            the interval between train and evaluation (default: 5)
  --data_dir                        directory of dataset (default: path/data)
  --eval_dir                        path of checkpoint files for evaluation (default: path/eval)
  --dtype                           data type of the inputs of the network (default: tf.float32)
  --use_nesterov                    whether to use Nesterov in momentum optimizer (dafault: True)
  --label_smoothing                 use label smooth in cross entropy (default 0.1)
  --weight_decay                    weight decay for regularization loss (default: 0.0001)
  --batch_size                      batch size per npu (default: 32)
  --lr                              initial learning rate (default: 0.01)
  --T_max                           T_max value in cosine annealing learning rate (default: 150)
  --momentum                        momentum value used in optimizer (default: 0.9)
  --display_every                   frequency to display infomation (default: 1)
  --log_name                        name of log file (default: vgg16.log)
  --log_dir                         path to save checkpoint and log (default: ./model_1p)  
```

### Training process

All the results of the training will be stored in the directory `log_dir`.
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |    Top1/Top5   |
| :--------: | :------------: |
|    150     | 73.986%/91.75% |

#### Training performance results
| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |     850 img/s     |

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |     4200 img/s    |











