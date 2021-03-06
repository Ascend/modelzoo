# EfficientNet-B0 for Tensorflow 

This repository provides scripts and recipe to train the EfficientNet-B0 model to achieve state-of-the-art accuracy.

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
  * [Data augmentation](#Data-augmentation)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Check json](#Check-json)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#Training-process)    
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options) 
​     

## Description

EfficientNets are a family of image classification models developed based on AutoML and Compound Scaling. EfficientNet-B0 is the mobile-size baseline network in this family. It is mainly constructed based on mobile inverted bottleneck blocks with squeeze-and-excitation optimization.

- EfficientNet-B0 model from: [Mingxing Tan and Quoc V. Le.  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019]<https://arxiv.org/abs/1905.11946>. 
- reference implementation: <https://github.com/tensorflow/tpu/tree/r1.15/models/official/efficientnet>.

## Requirements

- Tensorflow 1.15.0
- Download and preprocess ImageNet2012,CIFAR10 or Flower dataset for training and evaluation.


## Default configuration

The following sections introduce the default configurations and hyperparameters for EfficientNet-B0 model.

### Optimizer

This model uses RMSProp optimizer from Tensorflow with the following hyperparameters (for 8 NPUs):

- Momentum : 0.9
- Learning rate (LR) : 0.2
- LR schedule: exponential decay
- Warmup epoch: 5
- decay : 0.9
- epsilon : 0.001
- Batch size : 256*8
- Weight decay :  1e-5
- Moving average decay: 0.9999
- Label smoothing = 0.1
- We train for:
  - 350 epochs for a standard training process using ImageNet2012
  
Users can find more detailed parameters from the source code.

### Data augmentation

This model uses the following data augmentation:

- For training:
  - DecodeAndRandomCrop
  - RandomFlipLeftRight with probability of 0.5
  - Reshape to [224, 224, 3]
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- For inference:
  - DecodeAndCenterCrop
  - Reshape to [224, 224, 3]
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

For more details, we refer readers to read the corresponding source code.


## Quick start guide

### Prepare the dataset

1. Download the ImageNet2012 dataset
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and validation tfrecord files are under the path/data directories.


### Docker container scene

- Compile image
```bash
cd efficientnet
docker build -t ascend-efficientnet .
```

- Start the container instance
```bash
bash efficientnet/docker_start.sh
```

Parameter Description:

```bash
#!/usr/bin/env bash
docker_image=$1 \   #Accept the first parameter as docker_image
data_dir=$2 \       #Accept the second parameter as the training data set path
model_dir=$3 \      #Accept the third parameter as the model execution path
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #The number of cards used by docker, currently using 0~7 cards
        --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #Training data set path
        -v ${model_dir}:${model_dir} \  #Model execution path
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image is the image name
        /bin/bash
```

After executing docker_start.sh with three parameters:
  - The generated docker_image
  - Dataset path
  - Model execution path
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```



### Check json
Unzip the file and switch to the `efficientnet` directory, and check whether there is an 8 card IP json configuration file "8p.json" in the root directory.
8P json configuration file content:

```
{"group_count": "1","group_list": 
                    [{"group_name": "worker","device_count": "8","instance_count": "1", "instance_list": 
                    [{"devices":                    
                                   [{"device_id":"0","device_ip":"192.168.100.101"},
                                    {"device_id":"1","device_ip":"192.168.101.101"},
                                    {"device_id":"2","device_ip":"192.168.102.101"},
                                    {"device_id":"3","device_ip":"192.168.103.101"},
                                    {"device_id":"4","device_ip":"192.168.100.100"},      
                                    {"device_id":"5","device_ip":"192.168.101.100"},        
                                    {"device_id":"6","device_ip":"192.168.102.100"},     
                                    {"device_id":"7","device_ip":"192.168.103.100"}],      
                                    "pod_name":"ascend8p",        "server_id":"127.0.0.1"}]}],"status": "completed"}
```


### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

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
    - In *train_1p.sh*, python scripts part should look like as follows.Make sure that the "--data_dir" modify the path of the user generated tfrecord.

        ```shell
        python3.7 ${currentDir}/main_npu.py \
          --data_dir=/data/slimImagenet \
          --model_dir=./ \
          --mode=train \
          --train_batch_size=256 \
          --train_steps=100 \
          --iterations_per_loop=10 \
          --model_name=efficientnet-b0
        ```
    - Run the program  
        ```
        bash run_1p.sh
        ```
- Case for 8 NPUs
    - In *train_8p.sh*, python scripts part should look like as follows.Make sure that the "--data_dir" modify the path of the user generated tfrecord.
        ```shell 
        python3.7 ${currentDir}/main_npu.py \
          --data_dir=/data/slimImagenet \
          --model_dir=./ \
          --mode=train_and_eval \
          --train_batch_size=256 \
          --train_steps=218750 \
          --iterations_per_loop=625 \
          --steps_per_eval=31250 \
          --base_learning_rate=0.2 \
          --model_name=efficientnet-b0
        ```
    - Run the program  
        ```
        bash run_8p.sh
        ```

#### Training process

After training, all the results of the training will be stored in the directory `result`.

#### Evaluation

- In *test.sh*, python scripts part should look like as follows:
     ```shell 
    python3.7 ${currentDir}/main_npu.py \
      --data_dir=/data/slimImagenet \
      --mode=eval \
      --model_dir=result/8p/0/ \
      --eval_batch_size=128 \
      --model_name=efficientnet-b0
    ```
    Remember to modify the `data_dir` and `model_dir`, then run the program  
    ```
    bash test.sh
    ```

## Advanced

### Command-line options

We list those important parameters to train this network here. For more details of all the parameters, please read *main_npu.py* and other related files.

```
  --data_dir                        directory of dataset (default: FAKE_DATA_DIR)
  --model_dir                       directory where the model stored (default: None)
  --mode                            mode to run the code (default: train_and_eval)
  --train_batch_size                batch size for training (default: 2048)
  --train_steps                     max number of training steps (default: 218949)
  --iterations_per_loop             number of steps to run on device each iteration (default: 1251)
  --model_name                      name of the model (default: efficientnet-b0)
  --steps_per_eval                  controls how often evaluation is performed (default: 6255)
  --eval_batch_size                 batch size for evaluating in eval mode (default: 1024)
  --base_learning_rate              base learning rate for each card (default: 0.016)
```


 












