# DeepFM for TensorFlow

This repository provides a script and recipe to train the DeepFM model .

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Modification of OPP operator](#Modification-of-OPP-operator)
  * [Check json](#Check-json)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#Training-process)
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options)



## Description

DeepFM "DeepFM: A Factorization-Machine based Neural Network for CTR prediction" is an article published by Huawei in 2017. The existing solutions to the CTR estimation problem generally have a preference for different levels of cross-features, or require expert-level feature engineering. The article proposes the DeepFM model, which can achieve end-to-end training without additional feature engineering, and can automatically extract cross-features.

- DeepFM model from: <https://arxiv.org/abs/1703.04247>
- reference implementation: <https://github.com/ChenglongChen/tensorflow-DeepFM>

## Requirements

- Download and preprocess criteo dataset for training and evaluation.

## Quick start guide

### Prepare the dataset

- Download the dataset.

- The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

- Please convert the dataset to tfrecord format file by yourself.


### Docker container scene

- Compile image
```bash
docker build -t ascend-deepfm .
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


### Modification of OPP operator
After the environment run package is installed, set the`${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/config/${chip_info}/aic-${chip_info}-ops-info.json` The information of `sigmoid` operator and `reducesumd` operator is modified according to table 1. Where `${LOCAL_HIAI}` is `LOCAL_HIAI` is the installation location of run package, such as `/usr/local/ascend`, `${chip_info}` is the chip version, such as `ascend910`. Please modify it according to the actual situation.

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
### Check json
Check whether there is a JSON configuration file "8p.json" for 8 Card IP in the `scripts/directory`.
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

Configure data source and training steps:

- take `config.py` "--record_path" to the actual data path.

- take `config.py` "--n_epoches" to the number of epochs to be trained. 

The training data of each epoch in the script `train.py` is 1 / 5 of the total dataset, which can not be divided by 5. At this time, the total training epoch can be reduced by 5 times.

Script parameters:
```
--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128
--n_epoches                       initial learning rate,default: 0.06
```


### Running the example

#### Training

Start single card (1 chip) training:
1P:
```
cd scripts
bash run_npu_1p.sh
```
Start single card (8 chips) training:
8P:
```
cd scripts
bash run_npu_8p.sh
```

#### Training process

The storage path of the model is `results/1p` or `results/8p`, including the training log and checkpoints files. Take 8-card training as an example, the loss information is in the file `results/8p/0/model_in_8p/log`.

#### Evaluation

The evaluation process is followed by training. After each epoch training is completed, a round of evaluation is carried out.
The set path needs to be changed based on the actual path.
configs/config.py: record_path



## Advanced

### Command-line options

```
--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128 
--n_epoches                       initial learning rate,default: 0.06
--line_per_sample                 number of samples per line,make sure that it is divisible by batch_size
```



