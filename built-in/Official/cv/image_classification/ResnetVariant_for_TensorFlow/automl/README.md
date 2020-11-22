# ResNet Variant概述

ResNet是ImageNet竞赛中分类问题效果比较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。随后也有很多工作基于ResNet提出了变种网络结构，本次将基于ResNet结构进行神经网络架构搜索。

> 参考论文：[Kaiming He, Xiangyu Zhang, Shaoping Ren, Jian Sun. Deep Residual Learning for Image Recognition, CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf)

# 运行环境

- Python: >=3.6
- Vega: 1.0
- NPU芯片

# 配置环境变量

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息示例内容如下。

当版本为Atlas Data Center Solution V100R020C10时，请使用以下环境变量：

```bash
export install_path=/usr/local/Ascend/nnae/latest
# driver包依赖
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
#fwkacllib 包依赖
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
exportPATH=${install_path}/fwkacllib/ccec_compiler/bin:{install_path}/fwkacllib/bin:$PATH
#tfplugin 包依赖
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
# opp包依赖
export ASCEND_OPP_PATH=${install_path}/opp
```

# 安装Vega库

首先进入 automl 文件夹
```bash
cd /path/to/automl
```

automl文件夹下有一个 deploy 目录，该目录下的一个shell文件用于安装依赖库，运行如下命令：

```bash
bash deploy/install_dependencies.sh
```
将automl文件夹加入环境变量，这样才完成了vega库的安装

```bash
export PYTHONPATH=/path/to/automl/:$PYTHONPATH
```


# 代码目录结构

automl文件夹目录如下

```
├─benchmark
├─deploy                            // 部署代码
├─docs                              // 文档说明
├─evaluate_service
├─examples                          // 代码运行示例
│  ├─run_example.py                 // 示例代码运行入口
│  └─nas
│      └─backbone_nas               // ResNetVariant代码运行示例
│           ├─backbone_nas.yml      // Pytorch版本参数设置
│           └─backbone_nas_tf.yml   // TensorFlow版本参数设置
└─vega
    ├─algorithms                    // 内置算法
    ├─core                          // 核心部件
    ├─datasets                      // 数据集
    ├─model_zoo                     
    └─search_space                  // 搜索空间
```


# 数据集准备

首先下载 ImageNet 数据集，并且放到如下路径

```
/root/datasets/imagenet_tfrecord
```

你也可以放到其他位置，不过需要在yaml配置文件中修改对应的数据集路径




# 运行代码

automl下的examples目录提供了运行不同算法shell文件。你首先需要进入automl文件夹，之后你可以直接通过运行如下命令运行DARTS算法

```bash
cd /path/to/automl
bash ./examples/run_resnetvariant.sh 0
```

上面bash命令后的 0 用于指定设备ID，适用于单卡运行的场景。运行8卡时，你需要执行如下命令：

```bash
bash ./examples/run_resnetvariant_8p.sh 
```



# 实验结果

|Network|Network Type|Framework|Num of NPUs|Accuracy|Training Time|Atlas NPU Model|Server|Container|Precision|Dataset|Ascend AI Processor|NPU Version|备注|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ResNetVariant|CNN|Vega|1x|75.98|19.0 h|Atlas 800-9000|Atlas 800-9000|NA (Bare Metal)|Mixed|CIFAR-10|Ascend 910|Atlas 800-9000|-|
|ResNetVariant|CNN|Vega|8x|76.08|3.0 h|Atlas 800-9000|Atlas 800-9000|NA (Bare Metal)|Mixed|CIFAR-10|Ascend 910|Atlas 800-9000|-|


|Network | Network Type | Framework | Num of NPUs | Throughput | Batch Size | Atlas NPU Model | Server | Container | Precision | Dataset | Ascend AI Processor | NPU Version | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ResNetVariant | CNN | Vega | 1x | 1902 images/sec | 256 | Atlas800-9000 | Atlas800-9000 | NA (Bare Metal) | Mixed | ImageNet | Ascend 910 | Atlas800-9000 | - |
|ResNetVariant | CNN | Vega | 8x | 14336images/sec | 256 | Atlas800-9000 | Atlas800-9000 | NA (Bare Metal) | Mixed | ImageNet | Ascend 910 | Atlas800-9000 | - |



默认存储日志信息如下：

```
tasks/[task_id]/
    +-- output
    |       +-- (step name)
    |               +-- model_desc.json          # model desc, save_model_desc(model_desc, performance)
    |               +-- hyperparameters.json     # hyper-parameters, save_hps(hps), load_hps()
    |               +-- models                   # folder, save models, save_model(id, model, desc, performance)
    |                     +-- (worker_id).pth    # model file
    |                     +-- (worker_id).json   # model desc and performance file
    +-- workers
    |       +-- [step name]
    |       |       +-- [worker_id]
    |       |       +-- [worker_id]
    |       |               +-- logs
    |       |               +-- checkpoints
    |       |               |     +-- [epoch num].pth
    |       |               |     +-- model.pkl
    |       |               +-- result
    |       +-- [step name]
    +-- temp
    +-- visual
    +-- logs
```


# 参数配置

默认的数据集路径是 /root/datasets/imagenet_tfrecord，你可以在 /path/to/automl/examples/nas/backbone_nas/下的backbone_nas_tf.yml和backbone_nas_tf_8.yml 文件里修改：

```yml
fully_train:                # Full_train阶段的参数配置    
    dataset:                # 数据集配置        
        type: Imagenet        
            common:            
                data_path: /root/datasets/imagenet_tfrecord 
```