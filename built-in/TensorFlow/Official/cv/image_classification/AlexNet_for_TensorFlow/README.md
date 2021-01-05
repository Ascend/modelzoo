# Alexnet for Tensorflow 

This repository provides a script and recipe to train the AlexNet model .

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

## Description

- AlexNet model from: `Alex Krizhevsky. "One weird trick for parallelizing convolutional neural networks". <https://arxiv.org/abs/1404.5997>.`
- reference implementation: <https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html#alexnet>

## Requirements

- Tensorflow NPU environmemnt
- Download and preprocess ImageNet2012，CIFAR10 or Flower dataset for training and evaluation.

## Default configuration

The following sections introduce the default configurations and hyperparameters for AlexNet model.

### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.06
- LR schedule: cosine_annealing
- Batch size : 128*8 for 8 NPUs, 256 for single NPU 
- Weight decay :  0.0001. 
- Label smoothing = 0.1
- We train for:
  - 150 epochs ->  60.1% top1 accuracy

### Data augmentation

This model uses the following data augmentation:

- For training:
  - RandomResizeCrop, scale=(0.08, 1.0), ratio=(0.75, 1.333)
  - RandomHorizontalFlip, prob=0.5
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- For inference:
  - Resize to (256, 256)
  - CenterCrop to (224, 224)
  - Normalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

## Quick start guide

### Prepare the dataset

1. Please download the ImageNet dataset by yourself. 
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### Docker container scene

- Compile image
```bash
docker build -t ascend-alexnet .
```

- Start the container instance
```bash
bash scripts/docker_start.sh
```

- Parameter Description:
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

#### After executing docker_start.sh with three parameters:
- The generated docker_image
- Data set path
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

Single card training
--Setting single card training parameters(script in scripts/train_alexnet_1p.sh),examples are as follows.
Make sure that the "--data_dir" modify the path of the user generated tfrecord.
```
python3.7 ${EXEC_DIR}/train.py --rank_size=1 \
 --iterations_per_loop=100 \
 --batch_size=256 \
 --data_dir=/opt/npu/slimImagenet \
 --mode=train \
 --checkpoint_dir=${EXEC_DIR}/${RESULTS}/${device_id}/model_1p/ \
 --lr=0.015 \
 --log_dir=./model_1p > ./train_${device_id}.log 2>&1
```
Single card training instruction(Script in scripts/run_npu_1p.sh)
```
bash scripts/run_npu_1p.sh
```

8-card training
--Set 8 card training parameters(script in scripts/train_alexnet_8p.sh),examples are as follows.
Make sure that the "--data_dir" modify the path of the user generated tfrecord.

```
start_id=$((device_id*24))
end_id=$((device_id*24+23))
taskset -c ${start_id}-${end_id} python3.7 ${EXEC_DIR}/train.py --rank_size=8 \
                      --epochs_between_evals=1 \
                      --mode=train \
                      --max_epochs=150 \
	              --iterations_per_loop=100 \
	              --batch_size=128 \
	              --data_dir=/data/slimImagenet \
	              --lr=0.06 \
                      --checkpoint_dir=./model_8p \
	              --log_dir=./model_8p > ./train_${device_id}.log 2>&1
```
8-card training instruction(Script in scripts/run_npu_8p.sh)
```
bash run_npu_8p.sh
```
#### Training process

All the results of the training will be stored in the directory `results`.
Script will store:

 - checkpoints
 - log

#### Evaluation
After the 150 epoch training is completed, please refer to the test process in "get started quickly". You need to modify the script startup parameters(script in scripts/train_alexnet_8p.sh),set mode to evaluate,add checkpoints path
```“rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”replace with“#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”```
Then execute the script.
```
bash run_npu_8p.sh
```
If you want to output the validation results to the document description file, you need to modify the startup script parameters, otherwise output to the default log file.

## Advanced

### Command-line options

```
  --data_dir                        train data dir
  --num_classes                     num of classes in ImageNet（default:1000)
  --image_size                      image size of the dataset
  --batch_size                      mini-batch size (default: 128) per npu
  --pretrained                      path of pretrained model
  --lr                              initial learning rate(default: 0.06)
  --max_epochs                      max number of epoch to train the model(default: 150)
  --warmup_epochs                   warmup epoch(when batchsize is large)
  --weight_decay                    weight decay (default: 1e-4)
  --momentum                        momentum(default: 0.9)
  --label_smoothing                 use label smooth in CE, (default 0.1)
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



