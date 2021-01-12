# Wide&Deep for TensorFlow

## 目录


* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [准备数据集](#准备数据集)
  * [Docker容器场景](#Docker容器场景)
  * [OPP算子修改](#OPP算子修改)
  * [检查json](#检查json)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
    * [推理](#推理)
* [高级](#高级)
  * [脚本参数](#脚本参数) 


## 概述

Wide&Deep是一个同时具有Memorization和Generalization功能的CTR预估模型，该模型主要由广义线性模型（Wide网络）和深度神经网络（Deep网络）组成，对于推荐系统来说，Wide线性模型可以通过交叉特征转换来记忆稀疏特征之间的交互，Deep神经网络可以通过低维嵌入来泛化未出现的特征交互。与单一的线性模型（Wide-only）和深度模型（Deep-only）相比，Wide&Deep可以显著提高CTR预估的效果，从而提高APP的下载量。

参考论文：Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.

参考实现：https://arxiv.org/abs/1606.07792

## 要求

- 安装有昇腾 AI处理器的硬件环境

- 下载并预处理criteo数据集以进行培训和评估

## 默认配置

- 超参配置

  batch_size：16000

  l2 lambda：9e-6

  Wide部分：

  - wide_w维度：[184965,1]，184965为数据集中特征的数目

  - wide_b维度：[1]

  - FTRL优化器：learning_rate：3.5e-2、 l1：3e-8、 l2：1e-6、 initial_accum：1.0

  Deep部分：

  - 每个特征转换成嵌入向量的维度：80

  - dense层每层的维度：[1024,512,256,128]

  - 激活函数：relu

  - drop_out的keep_prob：1.0

  - Adam优化器：learning_rate：3e-4、eps：9e-8、decay_rate:0.8、decay_steps:5

- 数据集信息

  数据集：Criteo（以Wide&Deep/Train为例，仅作为用户参考示例）

  数据规格：39id字段，39权重字段，1标签字段

  训练集大小：25G

  数据数目：41,257,595

  测试集大小：2.8G

  数据数目：4,582,977


## 快速上手

### 准备数据集

- 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集有criteo。

- 训练集和验证集图片分别位于train/和val/文件夹路径下，同一目录下的所有数据都有标签。

- 当前提供的训练脚本中，是以criteo数据集为例，训练过程中进行数据预处理操作，请用户自行将数据集封装为tfrecord格式，后续训练过程中进行数据预处理操作，请用户使用该脚本之前自行修改训练脚本中的数据集加载和预处理方法；在使用其他数据集时，视具体需求添加类似的模块。


### Docker容器场景

- 编译镜像
```bash
docker build -t ascend-widedeep .
```

- 启动容器实例
```bash
bash scripts/docker_start.sh
```

参数说明:

```
#!/usr/bin/env bash
docker_image=$1 \   #接受第一个参数作为docker_image
data_dir=$2 \       #接受第二个参数作为训练数据集路径
model_dir=$3 \      #接受第三个参数作为模型执行路径
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #docker使用卡数，当前使用0~7卡
 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #训练数据集路径
        -v ${model_dir}:${model_dir} \  #模型执行路径
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image为镜像名称
        /bin/bash
```

执行docker_start.sh后带三个参数：
  - 生成的docker_image
  - 数据集路径
  - 模型执行路径
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```



### OPP算子修改

在环境run包安装完成后，将 `${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/config/${chip_info}/aic-${chip_info}-ops-info.json` 中 `Sigmoid` 算子和`ReduceSumD`算子的信息按照 表1 进行修改。其中 `${LOCAL_HIAI}` 为`LOCAL_HIAI`为run包安装位置，例`/usr/local/Ascend`, `${chip_info}`为芯片版本，例`ascend910`，请根据实际情况进行修改。

修改前：
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

修改后：
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

### 检查json

检查 `scripts/` 目录下是否有存在8卡IP的json配置文件“8p.json”。
8P的json配置文件内容:

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

### 关键配置修改

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

配置数据源及训练步数：

- 将config.py中record_path修改为实际数据路径。

- 将config.py中n_epoches修改为需要训练epoch数。train.py脚本中每个epoch的训练数据为全量数据集的1/5，可不除5，此时总训练epoch可减少5倍。
'train_per_epoch': config.train_size/5,


### 运行示例

#### 训练

执行训练脚本

- 启动单卡（1个芯片）训练：

```
cd scripts

bash run_npu_1p.sh
```

- 启动单卡（8个芯片）训练：

```
cd scripts

bash run_npu_8p.sh
```

#### 推理

推理过程紧接着训练，每个epoch训练完成后进行一轮推理

## 高级

### 脚本参数

在`configs/config.py`中进行设置。

```
--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128 
--n_epoches                       initial learning rate,default: 0.06
```


