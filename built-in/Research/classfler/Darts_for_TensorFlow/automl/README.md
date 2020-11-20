# DARTS算法概述

DARTS(Differentiable ARchiTecture Search)是最早提出基于梯度下降算法实现神经网络架构搜索（Neural Architecture Search, NAS）的算法之一，其通过softmax函数将离散的搜索空间松弛化为连续的搜索空间，从而允许在架构搜索时使用梯度下降。在整个搜索过程中，DARTS交替优化网络权重和架构权重，并且还进一步探讨了使用二阶优化来代替一阶优化的提高性能的可能性。相比如早期基于强化学习和进化算法的NAS算法，DARTS可以在更短时间和更少计算资源的情况下找到类似甚至更好的网络架构。本次提供的是基于TensorFlow框架的DARTS实现。

>参考论文：[Hanxiao Liu, Karen Simonyan, Yiming Yang. DARTS: Differentiable Architecture Search, ICLR 2019](https://arxiv.org/abs/1806.09055)
>
>参考实现：https://github.com/quark0/darts

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
├─evaluate_service
├─examples                          // 代码运行示例
│  ├─run_example.py                 // 示例代码运行入口
│  └─nas
│      └─darts_cnn                  // DARTS代码运行示例
│           ├─darts.yml             // Pytorch版本参数设置
│           └─darts_tf.yml          // TensorFlow版本参数设置
└─vega
    ├─algorithms                    // 内置算法
    ├─core                          // 核心部件
    ├─datasets                      // 数据集
    ├─model_zoo                     
    └─search_space                  // 搜索空间
```

# 数据集准备

首先下载CIFAR10数据集，并且放到如下路径

```
root/datasets/cifar10/cifar-10-batches-bin
```

你也可以放到其他位置，不过需要在yaml配置文件中修改对应的数据集路径


# 运行代码

automl下的examples目录提供了运行不同算法shell文件。你首先需要进入automl文件夹，之后你可以直接通过运行如下命令运行 DARTS 算法

```bash
cd /path/to/automl
bash ./examples/run_darts.sh 0
```

上面bash命令后的 `0`用于指定硬件ID，适用于单卡情况。


# 实验结果

训练的精度数据

|Network | Network Type | Framework | Num of NPUs | Accuracy | Training Time | Atlas NPU Model | Server | Container | Precision | Dataset | Ascend AI Processor | NPU Version | 备注|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|DARTS | CNN | Vega | 1x | 94.36 | 66 h | Atlas 800-9000 | Atlas 800-9000 | NA (Bare Metal) | Mixed | CIFAR-10 | Ascend 910 | Atlas 800-9000 | - |


训练性能

|Network | Network Type | Framework | Num of NPUs | Throughput | Batch Size | Atlas NPU Model | Server | Container | Precision | Dataset | Ascend AI Processor | NPU Version | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|DARTS | CNN | Vega | 1x | 210 images/sec | 96 | Atlas800-9000 | Atlas800-9000 | NA (Bare Metal) | Mixed | CIFAR-10 | Ascend 910 | Atlas800-9000 | - |

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


# 参数配置介绍

默认的数据集路径是 /root/datasets/cifar10/cifar-10-batches-bin，你可以在 /path/to/automl/examples/nas/darts_cnn/darts_tf.yml 文件里修改


```yaml
fully_train:                # Full_train阶段的参数配置
    dataset:                # 数据集配置
    type: Cifar10
        common:
            data_path: /root/datasets/cifar10/cifar-10-batches-bin
```

