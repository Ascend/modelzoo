# Deeplabv3 for Tensorflow   

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
    * [Training process](#training-process)
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Commmand-line options](#Commmand-line-options)

## Description

This repository provides a script and recipe to train the Deeplabv3 model. The code is based on tensorflow/models/research/deeplab,
modifications are made to run on NPU
- Deeplabv3 model from: Liang-Chieh Chen et al. "Rethinking Atrous Convolution for Semantic Image Segmentation". <https://arxiv.org/abs/1706.05587>.
- reference implementation: <https://github.com/tensorflow/models/tree/master/research/deeplab>

## Requirements

- Tensorflow NPU environmemnt Pillow.
- Download and preprocess VOC dataset,COCO dataset or SBD dataset for training and evaluation.

## Default configuration

The following sections introduce the default configurations and hyperparameters for Deeplabv3 model. We reproduce two training setups 
on VOC_trainaug datasets, evaluate on four setups. 

For detailed hpyerparameters, please refer to corresponding scripts under directory `scripts/`

### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- LR schedule: cosine_annealing
- Batch size : 8 * 8   
- Weight decay :  0.0001. 

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

You can use any datasets as you wish. Here, we only use voc2012_trainaug dataset as an example to illustrate the data generation. 
 - download the voc2012 datasets and put it under the `datasets/` path.
 - check if `SegmentationClassAug.zip` exists under `datasets/`,if not, you can download Semantic Boundaries Dataset by yourself.
 - txt file named trainaug.txt containing all the seg_image filenames
 - put all three files under `datasets/` directory
 - go the datasets directory and run the script to create tfrecord file. tfrecord file will be saved under `dataset/pascal_voc_seg/tfrecord`
```
cd datasets
convert_voc2012_aug.sh
``` 

For other datasets, you need following three files.  Create a script similar to `convert_voc2012_aug.sh` and 
execute the script when you get all three file ready. 
 - Training pictures and their annotation files (voc-style segmentation annotation format)
 - txt file for all the seg_image filenames
 - Similar to the processing script of convert_voc2012_aug.sh


### Docker container scene

- Compile image
```bash
docker build -t ascend-deeplabv3 .
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

All the scripts to tick off the training are located under `scripts/`.  As there are two types of resnet_101 checkpoints exist, original version 
and a so-called beta version available, there are two sets of scripts one for each checkpoint. All the scripts that end with `beta` are meant for 
resnet_v1_101_beta checkpoint. All the scripts that start with `train` are configuration files for the training. You are free to choose either of them as initial checkpoint since they can achieve comparable performance.  

- For resnet_v1_101_beta, you can download by yourself.
- For resnet_v1_101, you can download by yourself.

Create `pretrained/` directory, after you download either checkpoint or both, un-compress the tarball file and put them under `pretrained/` directory.

For instance, to train the model with beta version checkpoints

Set single card training parameters (the script is located in `scripts/train_1p_s16_beta.sh`), the example is as follows, please refer to the script for details:

```
CKPT_NAME=${CURRENT_DIR}/pretrained/model.ckpt
PASCAL_DATASET=${CURRENT_DIR}/datasets/pascal_voc_seg/tfrecord
NUM_ITERATIONS=30000
```

Set 8 card training parameters (the script is located in `scripts/train_s16_r1_beta.sh`, `scripts/train_s16_r2_beta.sh`), the example (`scripts/train_s16_r1_beta.sh`) is as follows:

```
CKPT_NAME=${CURRENT_DIR}/pretrained/model.ckpt
PASCAL_DATASET=${CURRENT_DIR}/datasets/pascal_voc_seg/tfrecord
NUM_ITERATIONS=15000 
```

- with OS=16
    - train on single NPU 
    
        ```
         cd scripts
         ./run_1p_s16_beta.sh
        ```
    - train on 8 NPUs
        ```
         cd scripts
         ./run_s16_beta.sh
        ```
- with OS=8
    - train on single NPU 
    
        ```
         cd scripts
         ./run_1p_s8_beta.sh
        ```
    - train on 8 NPUs
        ```
         cd scripts
         ./run_s8_beta.sh
        ```

***Note***: As the time consumption of the training for single NPU is much higher than that of 8 NPUs, we mainly experiment training using 8 NPUs.

#### Training process

All the results of the training will be stored in the directory `results`.
Script will store:
 - checkpoints
 - log

#### Evaluation

Test results will be saved as log file under `eval/${DEVICE_ID}/resnet_101/training.log`. The value for DEVICE_ID is 
specified in the scripts

- for the model trained with OS=16
```
cd scripts
./test_1p_s16_beta.sh

```
- for the model trained with OS=8
```
cd scripts
./test_1p_s8_beta.sh

```
- for the model trained with OS=8, multi_scale inputs
```
cd scripts
./test_ms_beta.sh

```
- for the model trained with OS=8,multi_scale inputs, random horizontal flip 
```
cd scripts
./test_ms_flip_beta.sh

```

## Advanced

### Commmand-line options
```
  --tf_initial_checkpoint           path to checkpoint of pretrained resnet_v1_101, default None
  --model_variant                   the backbone of model, default mobilenet_v2
  --atrous_rate                     the rate for atrous conv, default [1]
  --train_split                     the split for the data, default train
  --dataset                         name of dataset, default pascal_voc_seg
  --dataset_dir                     train dataset directory, default None
  --train_batch_size                mini-batch size per npu, default 8 
  --base_learning_rate              initial learning rate
  --weight_decay                    weight decay factor, default: 4e-5
  --momentum                        momentum factor, default: 0.9
  --training_number_of_steps        the number of training steps , default 30000
  --learning_policy                 the lr scheduling policy, default poly
  --bias_multiplier                 the gradient scale factor for bias , default 2.0
  --mode                            the mode to run the program , default train
  --fine_tune_batch_norm            flag indicates whether to fine-tune Batch-norm parameters, default True
  --output_stride                   the ratio of input to output spatial resolution, default 16
  --iterations_per_loop             the number of training step done in device before parameter sent back to host, default 10
  --log_name                        the name of training log file, default training.log
```
for a complete list of options, please refer to `train_npu.py` and `common.py`


