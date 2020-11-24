# CARS算法概述

在不同的应用场景中，计算资源约束条件有所不同，很多现有的NAS方法一次搜索只能得到一个网络结构，无法满足差异化的约束条件需求。此外，尽管基于进化算法的NAS方法取得了不错的性能，但是每代样本需要重头训练来进行评估，极大影响了搜索效率。考虑到现有方法的不足，我们提出一种基于连续进化的多目标高效神经网络结构搜索方法（CARS: Continuous Evolution for Efficient Neural Architecture Search）。CARS维护一个最优模型解集，每次用解集中的模型来更新超网络中的参数。在每次进化算法迭代的过程中，子代的样本可以直接从超网络和父样本中直接继承参数，有效提高了进化效率。CARS一次搜索即可获得一系列不同大小和精度的模型，供用户根据实际应用中的资源约束来挑选相应的模型。

> 参考论文：[Zhaohui Yang, Yunhe Wang, Xinghao Chen, Boxin Shi, Chao Xu, Chunjing Xu, Qi Tian, Chang Xu.CARS: Continuous Evolution for Efficient Neural Architecture Search, CVPR 2020.](https://arxiv.org/abs/1806.09055)

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
│      └─cars                      // cars代码运行示例
│           ├─cars.yml             // Pytorch版本参数设置
│           └─cars_tf.yml          // TensorFlow版本参数设置
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

automl下的examples目录提供了运行不同算法shell文件。你首先需要进入automl文件夹，之后你可以直接通过运行如下命令运行CARS算法

```bash
cd /path/to/automl
bash ./examples/run_cars.sh 
```

# 实验结果

训练的精度数据

|Network | Network Type | Framework | Num of NPUs | Accuracy | Training Time | Atlas NPU Model | Server | Container | Precision | Dataset | Ascend AI Processor | NPU Version | 备注|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|CARS-A | CNN | Vega | 8x | 93.14 | 62 h | Atlas 800-9000 | Atlas 800-9000 | NA (Bare Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas 800-9000 | -|
|CARS-B | CNN | Vega | 8x | 94.45 | 62 h | Atlas 800-9000 | Atlas 800-9000 | NA (Bare Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas 800-9000 | -|
|CARS-C | CNN | Vega | 8x | 94.59 | 62 h | Atlas 800-9000 | Atlas 800-9000 | NA (Bare Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas 800-9000 | -|

训练性能

|Network | Network Type | Framework | Num of NPUs | Throughput | Batch Size | Atlas NPU Model | Server | Container | Precision | Dataset | Ascend AI Processor | NPU Version | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|CARS-A | CNN | Vega | 8x | 2288 images/sec | 96 | Atlas800-9000 | Atlas800-9000 | NA (Bare |Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas800-9000 | - |
|CARS-B | CNN | Vega | 8x | 1544 images/sec | 96 | Atlas800-9000 | Atlas800-9000 | NA (Bare |Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas800-9000 | - |
|CARS-C | CNN | Vega | 8x | 1352 images/sec | 96 | Atlas800-9000 | Atlas800-9000 | NA (Bare |Metal) | Mixed | CIFAR10 | Ascend 910 | Atlas800-9000 | - |


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

默认的数据集路径是 `/root/datasets/cifar10/cifar-10-batches-bin`，你可以在 `/path/to/automl/examples/nas/cars/cars_tf.yml` 文件里修改：

```yaml
fully_train:                # Full_train阶段的参数配置
    dataset:                # 数据集配置
    type: Cifar10
        common:
            data_path: /root/datasets/cifar10/cifar-10-batches-bin
```


