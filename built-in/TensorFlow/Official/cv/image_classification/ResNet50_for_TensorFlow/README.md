# ResNet50 for Tensorflow

This repository provides a script and recipe to train the ResNet50 model to achieve state-of-the-art accuracy.

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
  * [Data augmentation](#Data-augmentation)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#Training-process)    
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options) 
  * [Config file options](#Config-file-options) 

    

## Description

ResNet50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

This is ResNet50 V1.5 model. The difference between V1 and V1.5 is that, ResNet V1.5 set stride=2 at the 3x3 convolution layer for a bottleneck block where V1 set stride=2 at the first 1x1 convolution layer.ResNet50 builds on 4 residuals bottleneck block.

## Requirements
Ensure you have the following components:
  - Tensorflow
  - Hardware environment with the Ascend AI processor
  - Download and preprocess ImageNet2012, CIFAR10 or Flower dataset for training and evaluation.

## Default configuration

The following sections introduce the default configurations and hyperparameters for ResNet50 model.

### Optimizer

This model uses Momentum optimizer from tensorflow with the following hyperparameters:

- Momentum : 0.9
- Learning rate (LR) : 0.8
- LR schedule: cosine_annealing
- Batch size : 256
- Weight decay :  0.0001. We do not apply weight decay on all bias and Batch Norm trainable parameters (beta/gamma)
- Label smoothing = 0.1
- We train for:
  - 90 epochs -> configuration that reaches 76.9% top1 accuracy
  - 120 epochs -> 120 epochs is a standard for ResNet50

### Data augmentation

This model uses the following data augmentation:

- For training:
  - Resize to (224, 224)
  - Normalize, mean=(121, 115, 100), std=(70, 68, 71)
- For inference:
  - Resize to (224, 224)
  - CenterCrop, ratio=0.8
  - Normalize, mean=(121, 115, 100), std=(70, 68, 71)


## Quick start guide

### Prepare the dataset
1. Download the classification dataset, like ImageNet2012, CIFAR10, Flower and so on.
2. Please convert the dataset to tfrecord format file by yourself.
3. The train and validation tfrecord files are under the path/data directories.

### Docker container scene

- Compile image
```bash
docker build -t ascend-resnet .
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

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### Running the example
#### Training
The 1P training script is located in `scripts/train_1p.sh`, and each independent process of 8P training is located in `scripts/train_sample.sh`.
The configuration file is located in scr/configs, res50_256bs_1p.py is used to verify the functionality of the model. res50_256bs_1p_eval.py is used to verify the accuracy and performance of the model, 8p is the same, users can modify the configuration by themselves.

1P training:

Set parameters in train_1p.sh.

As default:
```shell
python3.7 ../src/mains/res50.py \
        --config_file=res50_256bs_1p \
        --max_train_steps=1000 \
        --iterations_per_loop=100 \
        --debug=True \
        --eval=False \
        --model_dir=${currentDir}/d_solution/ckpt${DEVICE_ID} > ${currentDir}/log/train_${device_id}.log 2>&1
```
More parameters are in --config_file under src/configs.
```shell
cd /path/to/Modelzoo_Resnet50_HC/scripts
bash train_1p.sh
```

8P training is similar to the former：
```shell
cd /path/to/Modelzoo_Resnet50_HC/scripts
bash train_8p.sh
```

#### Training process

All the results of the training will be stored in the directory specified with `--model_dir` argument.
Script will store:
 - d_solution.
 - log.

#### Evaluation

The configuration file is located in `src/configs`, and the file name is configured by "--config_file". After modifying the mode=evaluate in the configuration file, execute the training script.
```
  # ======= basic config ======= #
     'mode':'evaluate',             # "train","evaluate","train_and_evaluate"
     'epochs_between_evals': 4,     # used if mode is "train_and_evaluate"
```
1P test instruction (the script is located in `scripts/train_1p.sh`)

```
bash train_1p.sh
```

8P training instructions (the script is located in `scripts/train_8p.sh`)

```
bash train_8p.sh
```



## Advanced
### Command-line options
```
  --config_file           config file name
  --max_train_steps       max train steps
  --iterations_per_loop   interations per loop
  --debug=True            debug mode
  --eval=False            if evaluate after train
  --model_dir             directory of train model
```

### Config file options

```
  --mode                  mode of train, evaluate or train_and_evaluate
  --epochs_between_evals  epoch num between evaluates while training
  --data_url              train data dir
  --num_classes           num of classes in dataset（default:1000)
  --height                image height of the dataset
  --width                 image width of the dataset
  --batch_size            mini-batch size (default: 256) per gpu
  --lr_decay_mode         type of LR schedule: exponential, cosine_annealing
  --learning_rate_maximum initial learning rate
  --num_epochs            poch num to train the model 
  --warmup_epochs         warmup epoch(when batchsize is large)
  --weight_decay          weight decay (default: 1e-4)
  --momentum              momentum(default: 0.9)
  --label_smooth          whether to use label smooth in CE
  --label_smooth_factor   smooth strength of original one-hot
  --log_dir               path to save log
```


 












