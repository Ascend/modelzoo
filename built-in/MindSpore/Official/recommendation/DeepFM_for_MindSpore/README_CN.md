# 目录

<!-- TOC -->

- [目录](#目录)
  - [DeepFM 概述](#deepfm-概述)
  - [模型架构](#模型架构)
  - [数据集](#数据集)
  - [环境要求](#环境要求)
  - [快速入门](#快速入门)
  - [脚本说明](#脚本说明)
  - [脚本和样例代码](#脚本和样例代码)
  - [脚本参数](#脚本参数)
  - [准备数据集](#准备数据集)
    - [处理真实世界数据](#处理真实世界数据)
    - [生成和处理合成数据](#生成和处理合成数据)
  - [训练过程](#训练过程)
    - [训练](#训练)
    - [分布式训练](#分布式训练)
  - [评估过程](#评估过程)
    - [评估](#评估)
  - [模型描述](#模型描述)
  - [性能](#性能)
    - [评估性能](#评估性能)
    - [推理性能](#推理性能)
  - [随机情况说明](#随机情况说明)
  - [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

## DeepFM 概述

要想在推荐系统中实现最大点击率，学习用户行为背后复杂的特性交互十分重要。虽然已在这一领域取得很大进展，但高阶交互和低阶交互的方法差异明显，亟需专业的特征工程。本论文中,我们将会展示高阶和低阶交互的端到端学习模型的推导。本论文提出的模型 DeepFM，结合了推荐系统中因子分解机和新神经网络架构中的深度特征学习。

[论文] https://arxiv.org/abs/1703.04247. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

## 模型架构

DeepFM 由两部分组成。FM 部分是一个因子分解机，用于学习推荐的特征交互；深度学习部分是一个前馈神经网络，用于学习高阶特征交互。
FM 和深度学习部分拥有相同的输入原样特征向量，让 DeepFM 能从输入原样特征中同时学习低阶和高阶特征交互。

## 数据集

- [1] Criteo Dataset.

## 环境要求

- 硬件（Ascend 或 GPU）
  - 使用 Ascend 或 GPU 处理器准备硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过后，即可获得资源。
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
  - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

1. 克隆代码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/model_zoo/official/recommend/deepfm
```

2. 下载数据集。

> 请参考[1]获得下载链接。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. 使用此脚本预处理数据。处理过程可能需要一小时，生成的 MindRecord 数据存放在 data/mindrecord 路径下。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. 通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估：

- Ascend 处理器环境运行

  ```训练示例
  # 运行训练示例
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # 运行分布式训练示例
  sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json

  # 运行评估示例
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  在分布式训练中，JSON 格式的 HCCL 配置文件需要提前创建。

  具体操作，参见：

  <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools>.

- 在 GPU 上运行

  如在 GPU 上运行,请配置文件 src/config.py 中的`device_target`从 `Ascend`改为`GPU`。

  ```训练示例
  # 运行训练示例
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > ms_log/output.log 2>&1 &

  # 运行分布式训练示例
  sh scripts/run_distribute_train.sh 8 /dataset_path

  # 运行评估示例
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='GPU' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path/deepfm.ckpt
  ```

## 脚本说明

## 脚本和样例代码

```deepfm
.
└─deepfm
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend处理器或GPU上进行单机训练(单卡)
    ├─run_distribute_train.sh         # 在Ascend处理器上进行分布式训练(8卡)
    ├─run_distribute_train_gpu.sh     # 在GPU上进行分布式训练(8卡)
    └─run_eval.sh                     # 在Ascend处理器或GPU上进行评估
  ├─src
    ├─__init__.py                     # python init文件
    ├─config.py                       # 参数配置
    ├─callback.py                     # 定义回调功能
    ├─deepfm.py                       # DeepFM网络
    ├─preprocess_data.py              # 数据预处理
    ├─generate_synthetic_data.py      # 合成数据
    ├─dataset.py                      # 创建DeepFM数据集
  ├─train.py                          # 训练网络
  ├─eval.py                           # 评估网络
  ├─export.py                         # 模型导出
  ├─mindspore_hub_conf.py             # mindspore hub配置
  ├─README.md
```

## 脚本参数

在 config.py 中可以同时配置训练参数和评估参数。

- 训练参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path
  --ckpt_path CKPT_PATH
                        Checkpoint path
  --eval_file_name EVAL_FILE_NAME
                        Auc log file path. Default: "./auc.log"
  --loss_file_name LOSS_FILE_NAME
                        Loss log file path. Default: "./loss.log"
  --do_eval DO_EVAL     Do evaluation or not. Default: True
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

- 评估参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

## 准备数据集

### 处理真实世界数据

1. 下载数据集，并将其存放在某一路径下，例如./data/origin_data。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

> 从[1]获取下载链接。

2. 使用此脚本预处理数据。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

### 生成和处理合成数据

1. 以下命令将会生成 4000 万行点击数据，格式如下：

> "label\tdense_feature[0]\tdense_feature[1]...\tsparse_feature[0]\tsparse_feature[1]...".

```bash
mkdir -p syn_data/origin_data
python src/generate_synthetic_data.py --output_file=syn_data/origin_data/train.txt --number_examples=40000000 --dense_dim=13 --slot_dim=51 --vocabulary_size=2000000000 --random_slot_values=0
```

2. 预处理生成数据。

```bash
python src/preprocess_data.py --data_path=./syn_data/  --dense_dim=13 --slot_dim=51 --threshold=0 --train_line_count=40000000 --skip_id_convert=1
```

## 训练过程

### 训练

- Ascend 处理器上运行

  ```运行命令
  python trin.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='Ascend' \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  上述 python 命令将在后台运行,您可以通过`ms_log/output.log`文件查看结果。

  训练结束后, 您可在默认文件夹`./checkpoint`中找到检查点文件。AUC 值保存在 auc.log 中，损失值保存在 loss.log 文件中。

  ```运行结果
  2021-03-04 21:57:12 epoch: 1 step: 2582, loss is 0.4697781205177307
  2021-03-04 22:01:02 epoch: 2 step: 2582, loss is 0.46246230602264404
  ...
  ```

  模型检查点将会储存在当前路径。

- GPU 上运行
  待运行。

### 分布式训练

- Ascend 处理器上运行

  ```运行命令
  sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json
  ```

  上述 shell 脚本将在后台运行分布式训练。请在`log[X]/output.log`文件中查看结果。损失值保存在 loss.log 文件中。

- GPU 上运行
  待运行。

## 评估过程

### 评估

- Ascend 处理器上运行时评估数据集

  在运行以下命令之前，请检查用于评估的检查点路径。

  ```命令
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/deepfm.ckpt' \
    --device_target='Ascend' > ms_log/eval_output.log 2>&1 &
  OR
  sh scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/deepfm.ckpt
  ```

  上述 python 命令将在后台运行，请在 eval_output.log 路径下查看结果。准确率保存在 auc.log 文件中。

  ```结果
  {'result': {AUC: 0.8078456952530648, eval time: 21.29369044303894s}}
  ```

- 在 GPU 运行时评估数据集
  待运行。

## 模型描述

## 性能

### 评估性能

| 参数           | Ascend                                                                                               | GPU    |
| -------------- | ---------------------------------------------------------------------------------------------------- | ------ |
| 模型版本       | DeepFM                                                                                               | 待运行 |
| 资源           | Ascend 910;CPU 2.60GHz,192 核；内存：755G                                                            | 待运行 |
| 上传日期       | 2021-03-27                                                                                           | 待运行 |
| MindSpore 版本 | 1.1.1                                                                                                | 待运行 |
| 数据集         | [1]                                                                                                  | 待运行 |
| 训练参数       | epoch=15, batch_size=16000, lr=5e-4                                                                  | 待运行 |
| 优化器         | Adam                                                                                                 | 待运行 |
| 损失函数       | Sigmoid Cross Entropy With Logits                                                                    | 待运行 |
| 输出           | AUC                                                                                                  | 待运行 |
| 损失           | 0.44                                                                                                 | 待运行 |
| 速度           | 单卡：22.89 毫秒/步;                                                                                 | 待运行 |
| 总时长         | 单卡：18 分钟;                                                                                       | 待运行 |
| 参数(M)        | 16.5                                                                                                 | 待运行 |
| 微调检查点     | 216M (.ckpt 文件)                                                                                    | 待运行 |
| 脚本           | [DeepFM 脚本](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm) | 待运行 |

### 推理性能

| 参数           | Ascend            | GPU    |
| -------------- | ----------------- | ------ |
| 模型版本       | DeepFM            | 待运行 |
| 资源           | Ascend 910        | 待运行 |
| 上传日期       | 2021-03-27        | 待运行 |
| MindSpore 版本 | 1.1.1             | 待运行 |
| 数据集         | [1]               | 待运行 |
| batch_size     | 16000             | 待运行 |
| 输出           | AUC               | 待运行 |
| 准确率         | 单卡：80.78%;     | 待运行 |
| 推理模型       | 216M (.ckpt 文件) | 待运行 |

## 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout 算子。

## ModelZoo 主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
