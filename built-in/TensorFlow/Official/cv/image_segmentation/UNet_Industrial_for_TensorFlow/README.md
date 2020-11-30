# U-Net Industrial for TensorFlow

This repository provides an auto-mix-precision script and recipe to train U-Net Industrial.


## Table of Contents

- [U-Net Industrial Defect Segmentation for TensorFlow](#u-net-industrial-defect-segmentation-for-tensorflow)
  - [Table of Contents](#table-of-contents)
  - [Model overview](#model-overview)
    - [Default Configuration](#default-configuration)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [Quick Start Guide](#quick-start-guide)
    - [Clone the repository](#clone-the-repository)
    - [Download and preprocess the dataset: DAGM2007](#download-and-preprocess-the-dataset-dagm2007)
    - [Pre-compile](#pre-compile)
    - [Run training](#run-training)
    - [Run evaluation](#run-evaluation)
    - [Command line options for advanced usage](#command-line-options-for-advanced-usage)
      - [general](#general)
      - [model](#model)
  - [Results](#results)
        - [Training accuracy](#training-accuracy)
        - [Training performance](#training-performance)
      - [Inference performance results](#inference-performance-results)

## Model overview

This U-Net model is an implementation of `TinyUNet`, which is a modified and more efficient version of U-Net and often used in industrial scenario. 

TinyUNet contains an encoder and an decoder and outputs the segmentation results of input images.

### Default Configuration

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

## Setup

The following section list the requirements in order to start training the U-Net model
(only the `TinyUnet` model is proposed here).

### Requirements

Tensorflow 1.15.0

## Quick Start Guide

To train your model using mixed precision with tensor cores or using FP32, perform the following steps using the
default configuration of the U-Net model (only `TinyUNet` has been made available here) on the DAGM2007 dataset.

### Clone the repository

```bash
git clone XXX
cd ModelZoo_UNet_Industrial_TF/
```

### Download and preprocess the dataset: DAGM2007
The model is compatible with the datasets on tensorflow official website.

```bash
./download_and_preprocess_dagm2007.sh /path/to/dataset/directory/
```

**Important Information:** Some files of the dataset require an account to be downloaded, the script will invite you to download them manually and put them in the correct directory.

### Pre-compile

```bash
pip3 install dllogger/
```

### Run training  

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

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

#### 


### Run evaluation

Model evaluation on a checkpoint can be launched by running  one of the scripts in the `./scripts` directory
called `./scripts/eval.sh `.

* eval on a checkpoint
```bash
cd scripts/
./eval.sh <path to result checkpoint> <path to dataset> <DAGM2007 classID (1-10)>
```


### Command line options for advanced usage

To see the full list of available options and their descriptions, use the -h or --help command line option, for example:

```bash
python main.py --help
```

The following mandatory flags must be used to tune different aspects of the training:

#### general

`--exec_mode=train_and_evaluate` Which execution mode to run the model into.

`--iter_unit=batch` Will the model be run for X batches or X epochs ?

`--num_iter=2500` Number of iterations to run.

`--batch_size=16` Size of each minibatch per GPU.

`--results_dir=/path/to/results` Directory in which to write training logs, summaries and checkpoints.

`--data_dir=/path/to/dataset` Directory which contains the DAGM2007 dataset.

`--dataset_name=DAGM2007` Name of the dataset used in this run (only DAGM2007 is supported atm).

`--dataset_classID=1` ClassID to consider to train or evaluate the network (used for DAGM).



