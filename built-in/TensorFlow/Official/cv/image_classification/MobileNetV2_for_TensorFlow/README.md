# MobileNetv2 for Tensorflow 

This repository provides a script and recipe to train the MobileNetv2 model to achieve state-of-the-art accuracy.

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
  * [Data augmentation](#Data-augmentation)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Key configuration changes](#Key-configuration-changes)
  * [Check json](#Check-json)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#Training-process)    
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options) 

    

## Description

MobileNetv2 is a mobile architecture. It is mainly constructed based on depthwise separable convolutions, linear bottlenecks and inverted residuals.

MobileNetv2 model from: [Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." CVPR 2018.](https://arxiv.org/abs/1801.04381)

## Requirements

- Tensorflow 1.15.0
- Download and preprocess ImageNet2012 or CIFAR10 dataset for training and evaluation.

## Default configuration

The following sections introduce the default configurations and hyperparameters for MobileNetv2 model.

### Optimizer

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

### Data augmentation

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


## Quick start guide

### Prepare the dataset

1. Download the ImageNet2012 dataset.
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and validation tfrecord files are under the path/data directories.

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### Check json

Check whether there is a JSON configuration file "8p.json" for 8 Card IP in the directory
8p configuration file example.

```
{
 "group_count": "1",
 "group_list": [
  {
   "group_name": "worker",
   "device_count": "8",
   "instance_count": "1",
   "instance_list": [
    {
     "devices":[
      {"device_id":"0","device_ip":"192.168.100.101"},
      {"device_id":"1","device_ip":"192.168.101.101"},
      {"device_id":"2","device_ip":"192.168.102.101"},
      {"device_id":"3","device_ip":"192.168.103.101"},
      {"device_id":"4","device_ip":"192.168.100.100"},
      {"device_id":"5","device_ip":"192.168.101.100"},
      {"device_id":"6","device_ip":"192.168.102.100"},
      {"device_id":"7","device_ip":"192.168.103.100"}
     ],
     "pod_name":"ascend8p",
     "server_id":"127.0.0.1"
    }
   ]
  }
 ],
 "status": "completed"
}
```

### Running the example

#### Training

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

#### Training process

All the results of the training will be stored in the directory `result`.

#### Evaluation
- We evaluate results by using following commands:
     ```shell 
    python3.7 eval_image_classifier_mobilenet.py --dataset_dir=/opt/npu/slimImagenet \
        --checkpoint_path=result/8p/0/results/model.ckpt-187500
    ```
    Remember to modify the dataset path and checkpoint path, then run the command.


## Advanced

### Command-line options

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


 











