# Training InceptionV4 for Tensorflow 

This repository provides a script and recipe to train the InceptionV4 model.

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

This is an implementation of the official InceptionV4 network as in Google's ModelZoo, written in Tensorflow 1.15.0 and run on Ascend 910.
InceptionV4 model from: [Christian Szegedy, Sergey loffe, Vincent Vanhoucke, Alex Alemi. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".](https://arxiv.org/abs/1602.07261).

## Requirements

- Tensorflow 1.15.0
- Download and preprocess ImageNet2012,CIFAR10 or Flower dataset for training and evaluation.

## Default configuration

The following sections introduce the default configurations and hyperparameters for InceptionV4 model.

### Optimizer

This model uses the RMSprop optimizer with the following hyperparameters:

- Momentum: 0.9
- Decay: 0.9
- epsilon: 1.0
- Origin learning rate: 0.045
- LR schedule: cosine_annealing
- Batch size : 1P (128), 8P (64 * 8) 
- Weight decay :  0.00001. 
- Label smoothing = 0.1
- We train for:
  - 100 epochs for a standard training process using official TFRecord dataset of ImageNet2012

### Data augmentation

This model uses the following data augmentation:

- For training:
  - Resized: (299, 299, 3)
  - RandomResizeCrop: scale=(0.05, 1.0), ratio=(0.75, 1.33)
  - Distort colour: random brightness, random saturation, random hue, random contrast
  - Normalized to [-1, 1]
- For inference:
  - Resized: (299, 299, 3)
  - CenterCrop
  - Normalized to [-1, 1]

## Quick start guide

### Prepare the dataset

1. Download the ImageNet2012 dataset.
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and validation tfrecord files are under the path/data directories.


### Docker container scene

- Compile image
```bash
docker build -t ascend-inceptionv4 .
```

- Start the container instance
```bash
bash scripts/docker_start.sh
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

Check whether there is a JSON configuration file "8p.json" for 8 Card IP in the scripts/ directory.
The content of the 8p configuration file:
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
                                    "pod_name":"npu8p",        "server_id":"127.0.0.1"}]}],"status": "completed"}
```

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### Running the example

#### Training

- train on a single NPU
    - **edit** *scripts/run_1p.sh* and *scripts/train_1p.sh* (see example below)
    - ./run_1p.sh
- train on 8 NPUs
    - **edit** *scripts/run_8p.sh* and *scripts/train_8p.sh* (see example below)
    - ./run_8p.sh 


Examples:

- Case for single NPU
    - Modify the ID of NPU in *device_group* in *scripts/run_1p.sh*, default ID is *0*.
    - In *scripts/train_1p.sh* , python scripts part should look like as follows. Make sure that the "--data_dir" modify the path of the user generated tfrecord.
        ```shell
        python3.7 ${dname}/train.py --rank_size=1 \
        --mode=train \
        --max_epochs=100 \
        --T_max=100 \
        --data_dir=/opt/npu/imagenet_data \
        --iterations_per_loop=10 \
        --batch_size=128 \
        --lr=0.045 \
        --display_every=100 \
        --log_dir=./model \
        --log_name=inception_v4.log > ${currentDir}/result/1p/train_${device_id}.log 2>&1 
        ```
    - Run the program  
        ```
        ./run_1p.sh
        ```
- Case for 8 NPUs
    - Modify the ID of NPU in *device_group* in *scripts/run_8p.sh*, default ID is *0,1,2,3,4,5,6,7*.
    - In *scripts/train_8p.sh* , python scripts part should look like as follows.Make sure that the "--data_dir" modify the path of the user generated tfrecord.
        ```shell 
        python3.7 ${dname}/train.py --rank_size=8 \
        --mode=train \
        --max_epochs=100 \
        --T_max=100 \
        --data_dir=/opt/npu/imagenet_data \
        --iterations_per_loop=10 \
        --batch_size=64 \
        --lr=0.045 \
        --display_every=100 \
        --log_dir=./model \
        --log_name=inception_v4.log > ${currentDir}/result/1p/train_${device_id}.log 2>&1 
        ```
    - Run the program  
        ```
        ./run_8p.sh
        ```

#### Training process

All the results of the training will be stored in the directory results. Script will store:

- checkpoints
- log

#### Evaluation
- Same procedure as training except 2 following modifications
    - change `--mode=train` to `--mode=evaluate`
    - add `--eval_dir=path/eval`
     ```shell 
    python3.7 ${dname}/train.py --rank_size=1 \
        --mode=evaluate \
        --data_dir=/opt/npu/imagenet_data \
        --batch_size=128 \
        --log_dir=./model \
        --eval_dir=./model \
        --log_name=eval_inceptionv4.log > ${currentDir}/result/1p/eval_${device_id}.log 2>&1 
    ```
    run the program  
    ```
    ./test.sh
    ```


## Advanced

### Command-line options

```
  --rank_size                       number of NPUs to use (default: 1)
  --mode                            mode to run the program (default: train_and_evaluate)
  --max_train_steps                 max number of training steps (default: 100)
  --iterations_per_loop             number of steps to run in each iteration (default: 10)
  --max_epochs                      number of epochs to train the model (default: 200)
  --epochs_between_evals            the interval between train and evaluation (default: 1)
  --data_dir                        directory of dataset (default: path/data)
  --eval_dir                        path of checkpoint files for evaluation (default: path/eval)
  --dtype                           data type of the inputs of the network (default: tf.float32)
  --use_nesterov                    whether to use Nesterov in momentum (dafault: True)
  --label_smoothing                 use label smooth in cross entropy (default 0.1)
  --weight_decay                    weight decay for regularization loss (default: 0.0004)
  --batch_size                      batch size per npu (default: 256)
  --lr                              initial learning rate (default: 0.01)
  --T_max                           T_max value in cosine annealing learning rate (default: 200)
  --momentum                        momentum value used in optimizer (default: 0.9)
  --display_every                   frequency to display infomation (default: 100)
  --log_name                        name of log file (default: inception_v4.log)
  --log_dir                         path to save checkpoint and log (default: ./model)  
```




