# U-Net Industrial for TensorFlow

This repository provides an auto-mix-precision script and recipe to train U-Net Industrial.


## Table of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Pre-compile](#Pre-compile)
  * [Key configuration changes](#Key-configuration-changes)
  * [Modification of OPP operator](#Modification-of-OPP-operator)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Commmand-line options](#Commmand-line-options)

## Description

This U-Net model is an implementation of `TinyUNet`, which is a modified and more efficient version of U-Net and often used in industrial scenario. 
TinyUNet contains an encoder and an decoder and outputs the segmentation results of input images.
- U-Net model from: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
- reference implementation: <https://ngc.nvidia.com/catalog/resources/nvidia:unet_industrial_for_tensorflow/files?version=2>

## Requirements

- Tensorflow 1.15.0
- Download and preprocess DAGM2007 dataset for training and evaluation.

## Default configuration

* Global Batch Size: 16
* Optimizer RMSProp:
    * decay: 0.9
    * momentum: 0.8
    * centered: True

* Learning Rate Schedule: Exponential Step Decay
    * decay: 0.8
    * steps: 500
    * initial learning rate: 1e-4

* Weight Initialization: He Uniform Distribution (introduced by [Kaiming He et al. in 2015](https://arxiv.org/abs/1502.01852) to address issues related ReLU activations in deep neural networks)

* Loss Function: Adaptive
    * When DICE Loss < 0.3, Loss = Binary Cross Entropy
    * Else, Loss = DICE Loss

* Data Augmentation
    * Random Horizontal Flip (50% chance)
    * Random Rotation 90°

* Activation Functions:
    * ReLU is used for all layers
    * Sigmoid is used at the output to ensure that the ouputs are between [0, 1]

* Weight decay: 1e-5

## Quick start guide

To train your model using mixed precision with tensor cores or using FP32, perform the following steps using the
default configuration of the U-Net model (only `TinyUNet` has been made available here) on the DAGM2007 dataset.

### Prepare the dataset

Download and preprocess the dataset: DAGM2007
The model is compatible with the datasets on tensorflow official website.

```bash
./download_and_preprocess_dagm2007.sh /path/to/dataset/directory/
```

**Important Information:** Some files of the dataset require an account to be downloaded, the script will invite you to download them manually and put them in the correct directory.

### Docker container scene

- Compile image
```bash
docker build -t ascend-unet .
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

- After executing docker_start.sh with three parameters:
The generated docker_image
Data set path
Model execution path
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```

### Pre-compile

```bash
pip3 install dllogger/
```

### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

### Modification of OPP operator

After the environment run package is installed, set the`${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/config/${chip_info}/aic-${chip_info}-ops-info.json` The information of `sigmoid` operator and `reducesumd` operator is modified according to table 1. Where` ${LOCAL_HIAI}` is `LOCAL_HIAI` is the installation location of run package, such as `/usr/local/ascend`, `${chip_info}` is the chip version, such as `ascend910`. Please modify it according to the actual situation.

1.Sigmoid

--Before modification:
```
"Sigmoid":{
    "input0":{
        "dtype":"float16,float",
        "name":"x",
        "paramType":"required",
        "reshapeType":"NC"
    },
    "op":{
        "pattern":"formatAgnostic"
    },
    "output0":{
        "dtype":"float16,float",
        "name":"y",
        "paramType":"required",
        "reshapeType":"NC",
        "shape":"all"
    }
},
```
--After modification:
```
"Sigmoid":{
    "input0":{
        "dtype":"float16,float",
        "name":"x",
        "paramType":"required",
        "reshapeType":"NC"
    },
    "op":{
        "pattern":"formatAgnostic"
    },
    "output0":{
        "dtype":"float16,float",
        "name":"y",
        "paramType":"required",
        "reshapeType":"NC",
        "shape":"all"
     },
    "precision_reduce":{
        "flag":"false"
     }
},
```
2.ReduceSumD

--Before modification:
```
"ReduceSumD":{
    "attr":{
        "list":"axes,keep_dims"
    },
    "attr_axes":{
        "paramType":"required",
        "type":"listInt",
        "value":"all"
    },
    "attr_keep_dims":{
        "defaultValue":"false",
        "paramType":"optional",
        "type":"bool",
        "value":"all"
    },
    "dynamicShapeSupport":{
        "flag":"true"
    },
    "input0":{
        "dtype":"float16,float",
        "name":"x",
        "paramType":"required",
        "unknownshape_format":"ND,ND"
    },
    "op":{
        "pattern":"reduce"
    },
    "output0":{
        "dtype":"float16,float",
        "name":"y",
        "paramType":"required",
        "unknownshape_format":"ND,ND"
    }
},
```
--After modification:
```
"ReduceSumD":{
    "attr":{
        "list":"axes,keep_dims"
    },
    "attr_axes":{
        "paramType":"required",
        "type":"listInt",
        "value":"all"
    },
    "attr_keep_dims":{
        "defaultValue":"false",
        "paramType":"optional",
        "type":"bool",
        "value":"all"
    },
    "dynamicShapeSupport":{
        "flag":"true"
    },
    "input0":{
        "dtype":"float16,float",
        "name":"x",
        "paramType":"required",
        "unknownshape_format":"ND,ND"
    },
    "op":{
        "pattern":"reduce"
    },
    "output0":{
        "dtype":"float16,float",
        "name":"y",
        "paramType":"required",
        "unknownshape_format":"ND,ND"
    },
    "precision_reduce":{
        "flag":"false"
    }
},
```

### Running the example

#### Training

* train on a single NPU
```bash
cd scripts/
./run_1p.sh <path to result repository> <path to dataset> <DAGM2007 classID (1-10)>
```
* train on 8 NPUs
```bash
cd scripts/
./run_8p.sh <path to result repository> <path to dataset> <DAGM2007 classID (1-10)>
```

#### Evaluation

Model evaluation on a checkpoint can be launched by running  one of the scripts in the `./scripts` directory
called `./scripts/eval.sh `.

* eval on a checkpoint
```bash
cd scripts/
./eval.sh <path to result checkpoint> <path to dataset> <DAGM2007 classID (1-10)>
```

## Advanced

To see the full list of available options and their descriptions, use the -h or --help command line option, for example:

```bash
python main.py --help
```

The following mandatory flags must be used to tune different aspects of the training:

### Commmand-line options

```
--exec_mode=train_and_evaluate          Which execution mode to run the model into.
--iter_unit=batch                       Will the model be run for X batches or X epochs ?
--num_iter=2500                         Number of iterations to run.
--batch_size=16                         Size of each minibatch per GPU.
--results_dir=/path/to/results          Directory in which to write training logs, summaries and checkpoints.
--data_dir=/path/to/dataset             Directory which contains the DAGM2007 dataset.
--dataset_name=DAGM2007                 Name of the dataset used in this run (only DAGM2007 is supported atm).
--dataset_classID=1                     ClassID to consider to train or evaluate the network (used for DAGM).
```


