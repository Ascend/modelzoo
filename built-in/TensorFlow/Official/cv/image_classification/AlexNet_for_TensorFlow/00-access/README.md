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
- Batch size : 128*8 for 8 NPUs, 256 for single NPU 
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
cd  ModelZoo_AlexNet_TF_HARD
```

### 2. Download and preprocess the dataset

1. download the ImageNet dataset.The model is compatible with the datasets on tensorflow official website.
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

### check json
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
### Start training.
Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### 3. Train

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
### 4.Training process

Start single card or multi card training through the training instructions in "quick start". Single card and multi card support single card and eight card network training by running different scripts.
Set the "data_dir" in the training script(train_alexnet_1p.sh,train_alexnet_8p.sh) as the path of the training dataset.Please refer to the "quick start" example for the specific process.
The storage path of the model is results/1p or results/8p, including the training log and checkpoints files. Take 8-card training as an example, the loss information is in the file results/8p/0/model_8p/alexnet_training.log An example is as follows.
```
step:   700  epoch:  0.6  FPS:33558.6, loss: 6.754, total_loss: 7.786  lr:0.00671  batch_time:3.051382
step:   800  epoch:  0.6  FPS:33828.3, loss: 6.840, total_loss: 7.871  lr:0.00767  batch_time:3.027051
step:   900  epoch:  0.7  FPS:32321.9, loss: 6.785, total_loss: 7.814  lr:0.00863  batch_time:3.168133
step:  1000  epoch:  0.8  FPS:34254.1, loss: 6.777, total_loss: 7.805  lr:0.00959  batch_time:2.989423
step:  1100  epoch:  0.9  FPS:32974.6, loss: 6.770, total_loss: 7.795  lr:0.01055  batch_time:3.105422
step:  1200  epoch:  1.0  FPS:32637.9, loss: 6.715, total_loss: 7.738  lr:0.01151  batch_time:3.137457
step:  1300  epoch:  1.0  FPS:33125.1, loss: 6.629, total_loss: 7.650  lr:0.01247  batch_time:3.091308
step:  1400  epoch:  1.1  FPS:32611.2, loss: 6.574, total_loss: 7.593  lr:0.01343  batch_time:3.140028
step:  1500  epoch:  1.2  FPS:33074.4, loss: 6.516, total_loss: 7.531  lr:0.01439  batch_time:3.096053
step:  1600  epoch:  1.3  FPS:35500.1, loss: 6.504, total_loss: 7.517  lr:0.01535  batch_time:2.884497
step:  1700  epoch:  1.4  FPS:32006.8, loss: 6.348, total_loss: 7.358  lr:0.01631  batch_time:3.199324
step:  1800  epoch:  1.4  FPS:34176.2, loss: 6.363, total_loss: 7.371  lr:0.01727  batch_time:2.996241
```

### 5.Verification/reasoning process
After the 150 epoch training is completed, please refer to the test process in "get started quickly". You need to modify the script startup parameters(script in scripts/train_alexnet_8p.sh),set mode to evaluate,add checkpoints path
```“rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”replace with“#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*”```
Then execute the script.
```
bash run_npu_8p.sh
```
If you want to output the validation results to the document description file, you need to modify the startup script parameters, otherwise output to the default log file.
```
Evaluating
Validation dataset size: 49921
 step  epoch  top1    top5     loss   checkpoint_time(UTC)
 6300    1.0  18.502   39.33    4.78  2020-06-18 11:18:45
12600    10.0  29.946   54.90    3.99  2020-06-18 11:42:07
125200   100.0  53.015   77.11    2.91  2020-06-18 12:40:13
187700   150.0  60.120   82.06    2.57  2020-06-18 13:12:14
Finished evaluation
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









