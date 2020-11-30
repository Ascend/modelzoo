# Training InceptionV4 for Tensorflow 

This repository provides a script and recipe to train the InceptionV4 model.

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

In this repository, we implement InceptionV4 from paper [Christian Szegedy, Sergey loffe, Vincent Vanhoucke, Alex Alemi. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".](https://arxiv.org/abs/1602.07261).

This is an implementation of the official InceptionV4 network as in Google's ModelZoo, written in Tensorflow 1.15.0 and run on Ascend 910.

### Model architecture

The model architecture can be found from the reference paper.

### Default configuration

The following sections introduce the default configurations and hyperparameters for InceptionV4 model.

#### Optimizer

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

#### Data augmentation

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

## Setup
The following section lists the requirements to train the InceptionV4 network.
### Requirements

Tensorflow 1.15.0

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd ModelZoo_InceptionV4_TF_Atlas
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset.The model is compatible with the datasets on tensorflow official website.
2. Generate tfrecord files following [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).
3. The train and validation tfrecord files are under the path/data directories.

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

### 3. Train

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- train on a single NPU
    - **edit** *scripts/run_1p.sh* and *scripts/train_1p.sh* (see example below)
    - ./run_1p.sh
- train on 8 NPUs
    - **edit** *scripts/run_8p.sh* and *scripts/train_8p.sh* (see example below)
    - ./run_8p.sh 


Examples:

- Case for single NPU
    - Modify the ID of NPU in *device_group* in *scripts/run_1p.sh*, default ID is *0*.
    - In *scripts/train_1p.sh* , python scripts part should look like as follows. For more detailed command lines arguments, please refer to [Command line arguments](#command-line-arguments)
        ```shell
        python3.7 ${dname}/train.py --rank_size=1 \
            --mode=train \
            --max_epoches=100 \
            --T_max=100 \
            --iterations_per_loop=10 \
            --data_dir=/opt/npu/data_PATH \
            --batch_size=128 \
            --lr=0.045 \
            --display_every=10 \
            --log_dir=./model \
            --log_name=inception_v4.log
        ```
    - Run the program  
        ```
        ./run_1p.sh
        ```
- Case for 8 NPUs
    - Modify the ID of NPU in *device_group* in *scripts/run_8p.sh*, default ID is *0,1,2,3,4,5,6,7*.
    - In *scripts/train_8p.sh* , python scripts part should look like as follows.
        ```shell 
        python3.7 ${dname}/train.py --rank_size=8 \
            --mode=train \
            --max_epochs=100 \
            --T_max=100 \
            --iterations_per_loop=10 \
            --data_dir=/opt/npu/data_PATH \
            --lr=0.045 \
            --log_dir=./model \
            --eval_dir=./model \
            --log_name=inception_v4.log
        ```
    - Run the program  
        ```
        ./run_8p.sh
        ```

### 4. Test
- Same procedure as training except 2 following modifications
    - change `--mode=train` to `--mode=evaluate`
    - add `--eval_dir=path/eval`
     ```shell 
    python3.7 ${dname}/train.py --rank_size=1 \
        --mode=evaluate \
        --data_dir=/opt/npu/data_PATH \
        --eval_dir=${dname}/scripts/result/8p/0/model \
        --log_dir=./model \
        --eval_dir=./model \
        --log_name=eval_inceptionv4.log > eval.log
    ```
    run the program  
    ```
    ./test.sh
    ```


## Advanced
### Commmand-line options

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

### Training process

Start single card or multi card training by training instruction in "quick start". Single card and multi card can support single card and 8 card network training by running different scripts.
Set data_dir in training script （train_1p.sh,train_8p.sh） as the path of training data set. Please refer to the "quick start" example for the specific process.
The storage path of the model is “results/1p”or“results/8p”, including the training log and checkpoints files. Taking single card training as an example, the loss information is in the file results/1p/0/model/inception_v4.log, and the example is as follows.
```
step: 12100  epoch:  1.2  FPS:  469.5  loss: 4.676  total_loss: 5.051  lr:0.04499
step: 12200  epoch:  1.2  FPS:  469.6  loss: 4.922  total_loss: 5.297  lr:0.04499
step: 12300  epoch:  1.2  FPS:  469.6  loss: 4.953  total_loss: 5.328  lr:0.04499
step: 12400  epoch:  1.2  FPS:  469.7  loss: 4.758  total_loss: 5.133  lr:0.04499
step: 12500  epoch:  1.2  FPS:  469.6  loss: 4.957  total_loss: 5.332  lr:0.04499
step: 12600  epoch:  1.3  FPS:  469.5  loss: 4.594  total_loss: 4.969  lr:0.04499
step: 12700  epoch:  1.3  FPS:  469.6  loss: 4.707  total_loss: 5.082  lr:0.04499
step: 12800  epoch:  1.3  FPS:  469.6  loss: 4.574  total_loss: 4.950  lr:0.04499
step: 12900  epoch:  1.3  FPS:  469.5  loss: 4.809  total_loss: 5.184  lr:0.04499
step: 13000  epoch:  1.3  FPS:  469.7  loss: 4.664  total_loss: 5.040  lr:0.04499
step: 13100  epoch:  1.3  FPS:  469.6  loss: 4.555  total_loss: 4.930  lr:0.04499
step: 13200  epoch:  1.3  FPS:  469.6  loss: 4.703  total_loss: 5.079  lr:0.04499
step: 13300  epoch:  1.3  FPS:  469.6  loss: 4.543  total_loss: 4.919  lr:0.04499
step: 13400  epoch:  1.3  FPS:  469.7  loss: 4.738  total_loss: 5.114  lr:0.04499
step: 13500  epoch:  1.3  FPS:  469.6  loss: 4.707  total_loss: 5.083  lr:0.04499
step: 13600  epoch:  1.4  FPS:  469.6  loss: 4.793  total_loss: 5.169  lr:0.04499
step: 13700  epoch:  1.4  FPS:  469.6  loss: 4.520  total_loss: 4.895  lr:0.04499
step: 13800  epoch:  1.4  FPS:  469.6  loss: 4.672  total_loss: 5.048  lr:0.04499
step: 13900  epoch:  1.4  FPS:  469.6  loss: 4.562  total_loss: 4.939  lr:0.04499
step: 14000  epoch:  1.4  FPS:  469.6  loss: 4.742  total_loss: 5.118  lr:0.04499
step: 14100  epoch:  1.4  FPS:  469.5  loss: 4.555  total_loss: 4.931  lr:0.04499
```
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |    Top1/Top5   |
| :--------: | :------------: |
|    100     | 74.9 %/92.3 %  |

#### Training performance results
| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |     483 img/s    |

| **NPUs** | train performance |
| :------: | :---------------: |
|    8     |     3170 img/s    |











