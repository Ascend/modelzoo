# DenseNet-121 for Tensorflow 

This repository provides a script and recipe to train the DenseNet-121 model .

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker container scene)
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

DenseNet-121 is a classic image classification network that uses a Dense Block structure that is connected at each layer.To improve the model efficiency and reduce parameters, the BN-ReLU-Conv(1*1)-BN-ReLU-Conv(3*3)bottleneck layer is used and the 1*1 conv is used to limit the number of input channels at each layer in the Dense Block to 4k(k indicates the number of output channels at each layer).DenseNet can effectively mitigate gradient loss,facilitates feature transmission and reuse.
DenseNet-121 model from: Gao Huang,Zhuang Liu,Laurens van der Maaten,Kilian Q.Weinberger."Densely Connected Convolutional Networks."arXiv:1608.06993


## Requirements

- Download and preprocess ImageNet2012 or CIFAR10 dataset for training and evaluation.

## Default configuration

### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:
- Batch size: 32
- Momentum: 0.9
- LR scheduler: cosine
- Learning rate(LR): 0.1
- Weight decay: 0.0001
- Label smoothing: 0.1
- train epoch: 150

## Quick start guide

### Prepare the dataset

The model is compatible with the datasets on tensorflow official website.
1. download the ImageNet dataset.
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### Docker container scene

- Compile image
```bash
docker build -t ascend-densenet .
```

- Start the container instance
```bash
bash scripts/docker_start.sh
```

- Parameter Description:
```bash
#!/usr/bin/env bash
docker_image=$1 \#Accept the first parameter as docker_image
data_dir=$2 \#Accept the second parameter as the training data set path
model_dir=$3 \#Accept the third parameter as the model execution path
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #The number of cards used by docker, currently using 0~7 cards
 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #Training data set path
        -v ${model_dir}:${model_dir} \  #Model execution path
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \#docker_image is the image name
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
Modify the *.JSON configuration file in the scripts directory to configure the relevant hardware IP information.
8P rank_table json Sample configuration file.
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

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### Running the example 

#### Training
- Single card training process
Configure training parameters
First, in the script `scritps/train_1p`.sh,the training dataset path and other parameters are configured,make sure that the "--data_dir" modify the path of the user generated tfrecord.
```
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=train \
    --max_train_steps=100 \
    --iterations_per_loop=10 \
    --data_dir=/opt/npu/slimImagenet \
    --display_every=10 \
    --log_dir=./model_1p \
    --log_name=densenet121_1p.log
```
Start training.
Starter card training(the script is `scripts/run_1p.sh`）:
```
bash run_1p.sh
```
- 8-card training process
Configure training parameters
First, in the script `scritps/train_8p.sh`,the training data set path and other parameters are configured,make sure that the "--data_dir" modify the path of the user generated tfrecord.
```
python3.7 ${dname}/train.py --rank_size=8 \
    --mode=train_and_evaluate \
    --max_epochs=150 \
    --iterations_per_loop=5004 \
    --epochs_between_evals=5 \
    --data_dir=/opt/npu/slimImagenet \
    --lr=0.1 \
    --log_dir=./model_8p \
    --log_name=densenet121_8p.log
```
Start training.
Starter 8-card training(the script is `scripts/run_8p.sh`）:
```
bash run_8p.sh
```
#### Training process

All the results of the training will be stored in the directory `results`.
Script will store:

 - checkpoints
 - log

#### Evaluation
When testing, you need to modify the script startup parameters(script in `scripts/test.sh`).Configure mode to evaluate and set it in eval_dir configure the path of checkpoint file.
```
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=evaluate \
    --data_dir=/opt/npu/slimImagenet \
    --eval_dir=${dname}/scripts/result/8p/0/model_8p \
    --log_dir=./ \
    --log_name=eval_densenet121.log > eval.log
```
Test instruction(script in `scripts/test.sh`)
```
bash test.sh
```

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












