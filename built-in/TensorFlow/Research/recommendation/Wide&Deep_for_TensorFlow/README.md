**Wide&Deep for TensorFlow** 

 **简述** 

Wide&Deep是一个同时具有Memorization和Generalization功能的CTR预估模型，该模型主要由广义线性模型（Wide网络）和深度神经网络（Deep网络）组成，对于推荐系统来说，Wide线性模型可以通过交叉特征转换来记忆稀疏特征之间的交互，Deep神经网络可以通过低维嵌入来泛化未出现的特征交互。与单一的线性模型（Wide-only）和深度模型（Deep-only）相比，Wide&Deep可以显著提高CTR预估的效果，从而提高APP的下载量。

参考论文：Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.

参考实现：https://arxiv.org/abs/1606.07792

 **快速上手** 

下载数据集并配置路径。

1、下载数据集。
请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集有criteo。该模型兼容tensorflow官网上的数据集。
训练集和验证集图片分别位于train/和val/文件夹路径下，同一目录下的所有数据都有标签。
当前提供的训练脚本中，是以criteo数据集为例，训练过程中进行数据预处理操作，请用户参考Tensorflow-Slim将数据集封装为tfrecord格式，后续训练过程中进行数据预处理操作，请用户使用该脚本之前自行修改训练脚本中的数据集加载和预处理方法；在使用其他数据集时，视具体需求添加类似的模块。

2、解压数据集如：tar -xzvf dac.tar.gz
将txt格式转换为h5/tfrecord格式(可选)。
```
python3.7 tools/process_data.py --raw_data_path [] --output []，进行数据预处理，生成h5/tfrecord格式数据集，--raw_data_path为上一步数据解压路径，--output_path为h5数据生成路径
```
例如：
```
python3.7 process_data.py ./src_path/ ./h5_data/
```
3、在环境run包安装完成后

```将${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/config/${chip_info}/aic-${chip_info}-ops-info.json中Sigmoid 算子和ReduceSumD算子的信息按照 表1 进行修改。其中${LOCAL_HIAI}为LOCAL_HIAI为run包安装位置，例/usr/local/Ascend, ${chip_info}为芯片版本，例ascend910，请根据实际情况进行修改。```

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

4、检查scripts/ 目录下是否有存在8卡IP的json配置文件“8p.json”。
8P的json配置文件内容
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

5、开始训练

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

6、配置数据源及训练步数
将config.py中record_path修改为实际数据路径。
将config.py中n_epoches修改为需要训练epoch数。train.py脚本中每个epoch的训练数据为全量数据集的1/5，可不除5，此时总训练epoch可减少5倍。
'train_per_epoch': config.train_size/5,
执行训练脚本。
启动单卡（1个芯片）训练：
```
cd scripts

bash run_npu_1p.sh
```
启动单卡（8个芯片）训练：
```
cd scripts

bash run_npu_8p.sh
```

 **脚本参数** 

在configs/config.py中进行设置。
```
--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128 
--n_epoches                       initial learning rate,default: 0.06
```

 **训练过程** 

通过“快速上手”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。

训练脚本会在训练过程中，每个epoch训练步骤保存一次checkpoint，结果存储在results/8p/1/train_1.log目录中。

训练脚本同时会每个step打印一次当前loss值，以查看loss收敛情况，如下所示：
```
epoch   1/ 20 - batch     1: loss = 0.768723, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 4 days, 23:18:27
epoch   1/ 20 - batch     2: loss = 0.762948, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 2 days, 11:39:07
epoch   1/ 20 - batch     3: loss = 0.760846, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 1 day, 15:46:00
epoch   1/ 20 - batch     4: loss = 0.763592, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 1 day, 5:49:27
epoch   1/ 20 - batch     5: loss = 0.764613, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 23:51:31
epoch   1/ 20 - batch     6: loss = 0.762559, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 19:52:53
epoch   1/ 20 - batch     7: loss = 0.765194, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 17:02:27
epoch   1/ 20 - batch     8: loss = 0.581548, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 14:54:37
epoch   1/ 20 - batch     9: loss = 0.568878, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 13:15:11
epoch   1/ 20 - batch    10: loss = 0.576262, auc = 0.000000, device_id = 0 | elapsed : 0:00:13, ETA : 11:55:39
```
验证/推理过程
推理过程紧接着训练，每个epoch训练完成后进行一轮推理，推理过程如下：
```
avg ctr on p: 0.256084  eval auc: 0.784103      log loss: 0.457939      ne: 0.804926    rig: 0.195074
avg ctr on p: 0.256084  eval auc: 0.792483      log loss: 0.453886      ne: 0.797801    rig: 0.202199
avg ctr on p: 0.256084  eval auc: 0.796951      log loss: 0.452282      ne: 0.794982    rig: 0.205018
avg ctr on p: 0.256084  eval auc: 0.799412      log loss: 0.451857      ne: 0.794236    rig: 0.205764
avg ctr on p: 0.256084  eval auc: 0.801143      log loss: 0.449913      ne: 0.793829    rig: 0.206171
avg ctr on p: 0.256084  eval auc: 0.802115      log loss: 0.449083      ne: 0.792358    rig: 0.207642
avg ctr on p: 0.256084  eval auc: 0.803227      log loss: 0.44791       ne: 0.792409    rig: 0.207591
avg ctr on p: 0.256083  eval auc: 0.804038      log loss: 0.44714       ne: 0.79149     rig: 0.20851
avg ctr on p: 0.256083  eval auc: 0.804702      log loss: 0.44674       ne: 0.792116    rig: 0.207884
avg ctr on p: 0.256084  eval auc: 0.80534       log loss: 0.445958      ne: 0.791039    rig: 0.208961
avg ctr on p: 0.256084  eval auc: 0.806138      log loss: 0.446081      ne: 0.79138     rig: 0.20862
avg ctr on p: 0.256084  eval auc: 0.806408      log loss: 0.445444      ne: 0.791385    rig: 0.208615
avg ctr on p: 0.256083  eval auc: 0.806843      log loss: 0.445112      ne: 0.790748    rig: 0.209252
avg ctr on p: 0.256081  eval auc: 0.806693      log loss: 0.444975      ne: 0.791508    rig: 0.208492
avg ctr on p: 0.256083  eval auc: 0.806821      log loss: 0.444531      ne: 0.79045     rig: 0.20955
avg ctr on p: 0.256083  eval auc: 0.80696       log loss: 0.444623      ne: 0.790902    rig: 0.209098
avg ctr on p: 0.256083  eval auc: 0.807251      log loss: 0.444793      ne: 0.791107    rig: 0.208893
avg ctr on p: 0.256083  eval auc: 0.807373      log loss: 0.444488      ne: 0.790668    rig: 0.209332
avg ctr on p: 0.256083  eval auc: 0.806936      log loss: 0.444224      ne: 0.790668    rig: 0.209332
```