# ResNet in TensorFlow On NPU
---

# Classification Model

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Check json](#Check-json)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)    
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options) 

## Description

1. ResNet is a relatively good network for classification problems in the ImageNet competition. It introduces the concept of residual learning. It protects the integrity of information by adding direct connection channels, solves problems such as information loss, gradient disappearance, and gradient explosion. The network can also be trained. ResNet has different network layers, commonly used are 18-layer, 34-layer, 50-layer, 101-layer, 152-layer. 

2. Ascend provides the V1.5 version of the 50-layer ResNet network this time. The difference between the V1.5 version of the ResNet network and the V1 version is that in the bottleneck module, the V1 version is set stride=2 in the first 1x1 convolutional layer, and V1.5 sets stride=2 in the 3x3 convolutional layer.


- This is an implementation of the ResNet101 model as described in the [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) paper.
 
- Current implementation is based on the code from the TensorFlow Official implementation in the [tensorflow/models Repo](https://github.com/tensorflow/models).

## Requirements

- Download and preprocess ImageNet2012，CIFAR10 or Flower dataset for training and evaluation.


## Default configuration

The following sections introduce the default configurations and hyperparameters for ResNet101 model.

### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Batch size: 128

- Momentum: 0.9

- LR scheduler: cosine

- Learning rate(LR): 0.05

- loss scale: 512

- Weight decay: 0.0001

- Label smoothing: 0.1

- train epoch: 90

## Quick start guide

### Prepare the dataset

- Please prepare the data set by yourself, including training set and validation set. The available data sets include ImageNet2012, CIFAR10, Flower, etc.

- The training set and validation set pictures are located in the "train/" and "val/" folder paths, and all pictures in the same directory have the same label.

- The currently provided training script takes the ImageNet2012 data set as an example. Data preprocessing is performed during the training process. Users are requested to modify the data set loading and preprocessing methods in the training script before using the script.


### Docker container scene

- Compile image
```bash
docker build -t ascend-resnet101 .
```

- Start the container instance
```bash
bash docker_start.sh
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

Modify the `*.json` configuration file in the `config` directory, modify the corresponding IP to the current IP, and change the board_id to the ID of the motherboard of the machine.

1P rank_table json configuration file:

```
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "1",
            "server_num": "1",
            "group_name": "",
            "instance_count": "1",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.100.101"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "0.0.0.0"
                }
           ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0"
    ],
    "para_plane_nic_num": "1",
    "status": "completed"
}
```

8P rank_table json configuration file:

```
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "",
            "instance_count": "8",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.100.101"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "1",
                            "device_ip": "192.168.101.101"
                        }
                    ],
                    "rank_id": "1",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "2",
                            "device_ip": "192.168.102.101"
                        }
                    ],
                    "rank_id": "2",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "3",
                            "device_ip": "192.168.103.101"
                        }
                    ],
                    "rank_id": "3",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "4",
                            "device_ip": "192.168.100.100"
                        }
                    ],
                    "rank_id": "4",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "5",
                            "device_ip": "192.168.101.100"
                        }
                    ],
                    "rank_id": "5",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "6",
                            "device_ip": "192.168.102.100"
                        }
                    ],
                    "rank_id": "6",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "7",
                            "device_ip": "192.168.103.100"
                        }
                    ],
                    "rank_id": "7",
                    "server_id": "0.0.0.0"
                }
            ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0",
        "eth1",
        "eth2",
        "eth3",
        "eth4",
        "eth5",
        "eth6",
        "eth7"
    ],
    "para_plane_nic_num": "8",
    "status": "completed"
}
```

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

check if path '/usr/local/HiAI' or ''/usr/local/Ascend' is existed or not.
modify '/usr/local/HiAI' to the actual path in scripts/run.sh

Before starting training, add the `ModelZoo_Resnet101_TF_Atlas` folder to the environment variable
```
export PYTHONPATH=/path/to/ModelZoo_Resnet101_TF_Atlas/:$PYTHONPATH
```


### Running the example

#### Training

During training, you need to modify the script startup parameters (the script is located in `official/r1/resnet/run_imagenet.sh` ) and set `eval_only` to `False`.Make sure that the "--data_dir" modify the path of the user generated tfrecord.

```
python3 $3/imagenet_main.py \
--resnet_size=101 \          # resnet的系列的模型类别，本例为resnet101，所以固定设置为101
--batch_size=128 \           # batchsize 
--num_gpus=1 \               # NPU设备数量，1P设置为1， 8P设置为8
--cosine_lr=True \           # 是否使用余弦的学习率策略
--dtype=fp16 \               # fp16 为设置为半精度训练，可与mixed精度配合使用
--label_smoothing=0.1 \      # label smooth的参数
--loss_scale=512 \           # 用于混合精度训练的loss scale
--train_epochs=90 \          # 总训练epoch数
--eval_only=True \         # 是否只做评估，训练时需要设置为False
--epochs_between_evals=10 \  # 每隔多少epoch做一次评估
--hooks=ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook \  # hook对象
--data_dir=/opt/npu/dataset/imagenet_TF_record \         # 数据集路径
--model_dir=./model_dir                                  # 模型、log等保存路径
```

`imagenet_main.py` is the Entry Python script.
`resnet_run_loop.py` is the Main Python script.

1P training instruction (the script is located in `official/r1/resnet/npu_train_1p.sh`)

```
bash npu_train_1p.sh
```

8P training instructions (the script is located in `official/r1/resnet/npu_train_8p.sh`)

```
bash npu_train_8p.sh
```

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.


#### Evaluation


When testing, you need to modify the script startup parameters (the script is located in `official/r1/resnet/run_imagenet.sh`) and set `eval_only` to `True`.Make sure that the "--data_dir" modify the path of the user generated tfrecord.

```
python3 $3/imagenet_main.py \
--resnet_size=101 \          # resnet的系列的模型类别，本例为resnet101，所以固定设置为101
--batch_size=128 \           # batchsize 
--num_gpus=1 \               # NPU设备数量，1P设置为1， 8P设置为8
--cosine_lr=True \           # 是否使用余弦的学习率策略
--dtype=fp16 \               # fp16 为设置为半精度训练，可与mixed精度配合使用
--label_smoothing=0.1 \      # label smooth的参数
--loss_scale=512 \           # 用于混合精度训练的loss scale
--train_epochs=90 \          # 总训练epoch数
--eval_only=True \         # 是否只做评估，训练时需要设置为False
--epochs_between_evals=10 \  # 每隔多少epoch做一次评估
--hooks=ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook \  # hook对象
--data_dir=/opt/npu/dataset/imagenet_TF_record \         # 数据集路径
--model_dir=./model_dir                                  # 模型、log等保存路径
```

1P test instruction (the script is located in `official/r1/resnet/npu_train_1p.sh`)

```
bash npu_train.1p.sh
```

8P test instruction (the script is located in `official/r1/resnet/npu_train_8p.sh`)

```
bash npu_train.8p.sh
```
 
## Advanced

### Command-line options

```
--resnet_size=101 \          # resnet的系列的模型类别，本例为resnet101，所以固定设置为101
--batch_size=128 \           # batchsize 
--num_gpus=1 \               # NPU设备数量，1P设置为1， 8P设置为8
--cosine_lr=True \           # 是否使用余弦的学习率策略
--dtype=fp16 \               # fp16 为设置为半精度训练,可与mixed精度配合使用
--label_smoothing=0.1 \      # label smooth的参数
--loss_scale=512 \           # 用于混合精度训练的loss scale
--train_epochs=90 \          # 总训练epoch数
--eval_only=True \         # 是否只做评估，训练时需要设置为False
--epochs_between_evals=10 \  # 每隔多少epoch做一次评估
--hooks=ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook \  # hook对象
--data_dir=/opt/npu/dataset/imagenet_TF_record \         # 数据集路径
--model_dir=./model_dir                                  # 模型、log等保存路径
```



    