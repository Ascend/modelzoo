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

1. Download the ImageNet2012 dataset
2. Generate tfrecord files following [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).
3. The train and validation tfrecord files are under the path/data directories.

### 3. Train
- train on a single NPU
    - **edit** *scripts/run_1p.sh* and *scripts/train_1p.sh* (see example below)
    - ./run_1p.sh
- train on 8 NPUs
    - **edit** *scripts/run_8p.sh* and *scripts/train_8p.sh* (see example below)
    - ./run_8p.sh 

Configure the env:
- **edit** *scripts/run_1p.sh* and *scripts/run_8p.sh*
    - add your own configuration info in *scripts/run_1p.sh* and *scripts/run_8p.sh*. The default configuration info could be considered as an example.
```shell
    #!/bin/bash
    install_path=/usr/local/Ascend
    export TBE_IMPL_PATH=${install_path}/nnae/latest/opp/op_impl/built-in/ai_core/tbeexport 
    LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:${install_path}/nnae/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons
    export PYTHONPATH=$PYTHONPATH:${install_path}/nnae/latest/opp/op_impl/built-in/ai_core/tbe:${install_path}/tfplugin/latest/tfplugin/python/site-packages/:${install_path}/nnae/latest/fwkacllib/python/site-packages/hccl:${install_path}/nnae/latest/fwkacllib/python/site-packages/te:${install_path}/nnae/latest/fwkacllib/python/site-packages/topi
    export PATH=${install_path}/nnae/latest/fwkacllib/ccec_compiler/bin:${PATH}
```

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

All the results and training ckpt models of the training will be stored in the directory `log_dir`.
 
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











