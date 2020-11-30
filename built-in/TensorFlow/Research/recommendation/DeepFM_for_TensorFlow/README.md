# DeepFM for TensorFlow

This repository provides a script and recipe to train the DeepFM model .

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



## Prepare the datasets

Download the dataset.The model is compatible with the datasets on tensorflow official website.
Users are requested to prepare their own data sets, including training set and verification set. The optional data set is criteo.
The images of training set and verification set are located in the path of train/ and val/ respectively, and all data in the same directory have labels.
In the current training scripts, the criteo dataset is taken as an example. Data preprocessing is performed during the training process. Please refer to tensorflow slim to encapsulate the data set into tfrecord format. For data preprocessing in the subsequent training process, users are requested to modify the data set loading and preprocessing methods in the training script before using the script; when using other data sets, the user can modify the data set loading and preprocessing methods in the training script Add similar modules according to specific requirements.


## Modification of OPP operator
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
## check json
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
## Start training.

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

## Configure data source and training steps.
-take config.py record_path to the actual data path.
-take config.py n_epoches are modified to the number of epochs to be trained. train.py The training data of each epoch in the script is 1 / 5 of the total data set, which can not be divided by 5. At this time, the total training epoch can be reduced by 5 times.


## Script parameters
```
--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128
--n_epoches                       initial learning rate,default: 0.06
```
## Execute training script.

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

## Training process.

The storage path of the model is `results/1p` or `results/8p`, including the training log and checkpoints files. Take 8-card training as an example, the loss information is in the file `results/8p/0/model_in_8p/log`, an example is shown below.
```
epoch   1/ 40 - batch     1: loss = 0.768723, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 4 days, 23:18:27
epoch   1/ 40 - batch     2: loss = 0.762948, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 2 days, 11:39:07
epoch   1/ 40 - batch     3: loss = 0.760846, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 1 day, 15:46:00
epoch   1/ 40 - batch     4: loss = 0.763592, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 1 day, 5:49:27
epoch   1/ 40 - batch     5: loss = 0.764613, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 23:51:31
epoch   1/ 40 - batch     6: loss = 0.762559, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 19:52:53
epoch   1/ 40 - batch     7: loss = 0.765194, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 17:02:27
epoch   1/ 40 - batch     8: loss = 0.581548, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 14:54:37
epoch   1/ 40 - batch     9: loss = 0.568878, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 13:15:11
epoch   1/ 40 - batch    10: loss = 0.576262, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 11:55:39
```

## Verification / reasoning process

The reasoning process is followed by training. After each epoch training, a round of reasoning is carried out. The reasoning process is as follows:
```
avg ctr on p: 0.256078  eval auc: 0.802047      log loss: 0.448934      ne: 0.789106    rig: 0.210894
avg ctr on p: 0.256078  eval auc: 0.805482      log loss: 0.447196      ne: 0.786052    rig: 0.213948
avg ctr on p: 0.256078  eval auc: 0.807624      log loss: 0.443865      ne: 0.780196    rig: 0.219804
avg ctr on p: 0.256078  eval auc: 0.809004      log loss: 0.442579      ne: 0.777936    rig: 0.222064
avg ctr on p: 0.256078  eval auc: 0.809839      log loss: 0.44191       ne: 0.776761    rig: 0.223239
avg ctr on p: 0.256078  eval auc: 0.810344      log loss: 0.441466      ne: 0.775979    rig: 0.224021
```

The set path needs to be changed based on the actual path.
configs/config.py: record_path

The result file will be saved in result file.

1P result:
    # dropout=0.8  eval auc 0.810(10epoch)

    model = FMNN_v2([input_dim, num_inputs, config.multi_hot_flags,
                   config.multi_hot_len],
                  [80, [1024, 512, 256, 128], 'relu'],
                  ['uniform', -0.01, 0.01, seeds[4:14], None],
                  ['adam', 5e-4, 5e-8, 0.95, 625],
                  [0.8, 8e-5],
                  _input_d
                  )
8P result:
   #dropout=0.8  eval auc 0.80788(15epoch)