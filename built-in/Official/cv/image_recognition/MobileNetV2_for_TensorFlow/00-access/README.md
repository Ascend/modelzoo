# MobileNetv2 for Tensorflow 

This repository provides a script and recipe to train the MobileNetv2 model to achieve state-of-the-art accuracy.

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

In this repository, we implement MobileNetv2 from paper [Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." CVPR 2018.](https://arxiv.org/abs/1801.04381)

MobileNetv2 is a mobile architecture. It is mainly constructed based on depthwise separable convolutions, linear bottlenecks and inverted residuals.

### Model architecture

The model architecture can be found from the reference paper.

### Default configuration

The following sections introduce the default configurations and hyperparameters for MobileNetv2 model.

#### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.8
- LR schedule: cosine_annealing
- Warmup epoch: 5
- Batch size : 256*8 
- Weight decay :  0.00004 
- Moving average decay: 0.9999
- Label smoothing = 0.1
- We train for:
  - 300 epochs for a standard training process using ImageNet2012

#### Data augmentation

This model uses the data augmentation from InceptionV2:

- For training:
  - Convert DataType and RandomResizeCrop
  - RandomHorizontalFlip, prob=0.5
  - Subtract with 0.5 and multiply with 2.0
- For inference:
  - Convert DataType
  - CenterCrop 87.5% of the original image and resize to (224, 224)
  - Subtract with 0.5 and multiply with 2.0

For more details, we refer readers to read the corresponding source code in Slim.

## Setup
The following section lists the requirements to start training the MobileNetv2 model.
### Requirements

Tensorflow 1.15.0

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd ModelZoo_MobileNetv2_TF_HARD
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset
2. Generate tfrecord files following [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).
3. The train and validation tfrecord files are under the path/data directories.

### 3. Set environment

Set environment variable like *LD_LIBRARY_PATH*, *PYTHONPATH* and *PATH* to match your system before training and testing.

### 4. Train
- train on a single NPU
    - **edit** *train_1p.sh* (see example below)
    - bash run_1p.sh
- train on 8 NPUs
    - **edit** *train_8p.sh* (see example below)
    - bash run_8p.sh 

Examples:
- Case for single NPU
    - In *train_1p.sh*, python scripts part should look like as follows. For more detailed command lines arguments, please refer to [Command line arguments](#command-line-arguments)
        ```shell
        python3.7 ${currentDir}/train.py \
            --dataset_dir=/opt/npu/slimImagenet \
            --max_train_steps=500 \
            --iterations_per_loop=50 \
            --model_name="mobilenet_v2" \
            --moving_average_decay=0.9999 \
            --label_smoothing=0.1 \
            --preprocessing_name="inception_v2" \
            --weight_decay='0.00004' \
            --batch_size=256 \
            --learning_rate_decay_type='cosine_annealing' \
            --learning_rate=0.4 \
            --optimizer='momentum' \
            --momentum='0.9' \
            --warmup_epochs=5
        ```
    - Run the program  
        ```
        bash run_1p.sh
        ```
- Case for 8 NPUs
    - In *train_8p.sh*, python scripts part should look like as follows.
        ```shell 
        python3.7 ${currentDir}/train.py \
            --dataset_dir=/opt/npu/slimImagenet \
            --max_epoch=300 \
            --model_name="mobilenet_v2" \
            --moving_average_decay=0.9999 \
            --label_smoothing=0.1 \
            --preprocessing_name="inception_v2" \
            --weight_decay='0.00004' \
            --batch_size=256 \
            --learning_rate_decay_type='cosine_annealing' \
            --learning_rate=0.8 \
            --optimizer='momentum' \
            --momentum='0.9' \
            --warmup_epochs=5
        ```
    - Run the program  
        ```
        bash run_8p.sh
        ```

### 5. Test
- We evaluate results by using following commands:
     ```shell 
    python3.7 eval_image_classifier_mobilenet.py --dataset_dir=/opt/npu/slimImagenet \
        --checkpoint_path=result/8p/0/results/model.ckpt-187500
    ```
    Remember to modify the dataset path and checkpoint path, then run the command.


## Advanced
### Commmand-line options

We list those important parameters to train this network here. For more details of all the parameters, please read *train.py* and other related files.

```
  --dataset_dir                     directory of dataset (default: /opt/npu/models/slimImagenet)
  --max_epoch                       number of epochs to train the model (default: None) 
  --max_train_steps                 max number of training steps (default: None)
  --iterations_per_loop             number of steps to run in devices each iteration (default: None)
  --model_name                      name of the model to train (default: mobilenet_v2_140)
  --moving_average_decay            the decay to use for the moving average (default: None)
  --label_smoothing                 use label smooth in cross entropy (default: 0.1)
  --preprocessing_name              preprocessing method for training (default: inception_v2)
  --weight_decay                    weight decay for regularization loss (default: 0)
  --batch_size                      batch size per npu (default: 96)
  --learning_rate_decay_type        learning rate decay type (default: fixed)
  --learning_rate                   initial learning rate (default: 0.1)
  --optimizer                       the name of optimizer (default: sgd)
  --momentum                        momentum value used in optimizer (default: 0.9)
  --warmup_epochs                   warmup epochs for learning rate (default: 5)
```

### Training process

All the results of the training will be stored in the directory `result`.
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |      Top1      |
| :--------: | :------------: |
|    300     |     72.47%     |

#### Training performance results
| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |     1400 img/s    |

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |    11000 img/s    |











