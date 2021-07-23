# DeepFM for CTR

## 1. 概述

DeepFM 模型结合了广度和深度模型的优点，联合训练 FM 模型和 DNN 模型，来同时学习低阶特征组合和高阶特征组合。该模型的 Deep component 和 FM component 从 Embedding 层共享数据输入，优势在于 Embedding 层的隐式向量在（残差反向传播）训练时可以同时接受到 Deep component 和 FM component 的信息，从而使 Embedding 层的信息表达更加准确而最终提升推荐效果。

DeepFM 网络的基本结构如下图。

![DeepFM_structure](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/DeepFM_structure.png)

## 2. 数据准备

### 2.1. 数据介绍

以[Criteo 数据集](https://www.kaggle.com/c/criteo-display-ad-challenge/data)为例。

Criteo 原始训练数据集为 train.txt, 测试集为 test.txt。 样例如下。

![demo_origin_data](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/demo_origin_data.png)

criteo 原始数据中，第一列为 label，表示是否点击，后续 39 列为特征列，其中前 13 列为连续值特征，后 26 列为离散类别特征（为经过 hash 处理后的字符串）。原始数据集需要经过数据预处理之后才能进行训练。

### 2.2. 数据预处理

数据预处理详情请参考[DeepFM 数据预处理](https://github.com/huaweicloud/ModelArts-Lab/tree/master/tools/DeepFM-GPU)，或者使用自己的预处理脚本(可忽略本文档预处理流程）, 但数据输出格式必须和此文档保持一致。

#### 2.2.1 预处理流程

- 统计各连续值列的 min-max value; 统计各类别列的词典频次字典；
- 按 threshold=100,对频次词典进行过滤，得到类别映射为 id 的 map 字典；
- 对连续值进行 MinMaxScaler; 将类别列特征映射为 id 值；
- 最后得到 feature 特征分为两种，id 特征 和 weights 特征 。
  - id 特征存储的值都是 map id，包括：
    - 连续特征列的 map id（本例连续特征共 13 列，map id 分别对应 0-12）
    - 离散特征列的 map id 值（26 列）
  - weights 特征包括：
    - 连续值 weights（MinMaxScaler 处理后的值）
    - 离散值 weights（有值的部分为 1，Nan(空值)的为 0）

处理后的特征数据保存为 MindRecord 文件，特征数据共 78 列(0-38 为 id 特征列，39-77 为 weights 特征列)，内容格式如下：
criteo feature 数据：
[![demo_input_features](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/demo_input_features.png)](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/demo_input_features.png)

criteo label 数据：

[![demo_output_label](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/demo_output_label.png)](https://cnnorth4-job-train-algorithm.obs.cn-north-4.myhuaweicloud.com/aimarket_deps/82cc6f9c-819e-4040-b033-10dfd7cac75b/pics/demo_output_label.png)

#### 2.2.2 数据输出格式

处理之后的数据格式如下：

1. `├─mindrecord`
2. `│ ├─train_input_part.mindrecord00`
3. `│ ├─train_input_part.mindrecord00.db`
4. `│ ├─train_input_part.mindrecord01`
5. `│…`
6. `│ ├─test_input_part.mindrecord0`
7. `│ ├─test_input_part.mindrecord0.db `
8. `│…`
9. `├─stats_dict`
10. `│ ├─cat_count_dict.pkl`
11. `│ ├─val_max_dict.pkl`
12. `│ ├─val_min_dict.pkl`

## 3. 训练

### 3.1. 算法基本信息

- 任务类型：推荐系统
- 支持的框架引擎：Ascend-Powered-Engine | Mindspore-1.1.1-python3.7-aarch64

### 3.2. 订阅算法

订阅算法流程请参考[使用 AI 市场的预置算法训练模型-订阅算法](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0025.html#modelarts_10_0025__section87421022184315)。

### 3.3. 创建训练作业

数据预处理完成之后，将预处理后的数据迁移至 OBS，上传 OBS 请参考[如何上传数据至 OBS？](https://support.huaweicloud.com/modelarts_faq/modelarts_05_0013.html) 。

**注意： 如果使用自定义预处理脚本，数据输出格式一定要和`2.2.2 数据输出格式`保持一致。**

使用订阅算法创建训练作业请参考 [使用 AI 市场的预置算法训练模型-创建训练作业](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0025.html#modelarts_10_0025__section139551128135716)。
区别在于，**数据来源**选择**数据存储位置**，选择上传的 OBS 路径即可，例如，上传的 OBS 路径为：/my-bucket/deepfm/dataset_path/(目录结构与`2.2.2 数据输出格式`保持一致)，则数据存储位置为/my-bucket/deepfm/dataset_path。
算法默认会去选择 mindrecord 文件夹下的数据，并加载 config.yaml 中的参数进行训练，stats_dict 文件夹用于适配在线推理服务。

### 3.4. 训练参数说明

|      名称       |  默认值  |  类型   | 是否必填 |               描述               |
| :-------------: | :------: | :-----: | :------: | :------------------------------: |
|   batch_size    |  16000   |   int   |   True   |   一次训练所抓取的数据样本数量   |
|  train_epochs   |    15    |   int   |   True   |             训练轮数             |
|  learning_rate  |   5e-4   |  float  |   True   |              学习率              |
|     resume      |    ""    | string  |  False   |         断点训练模型文件         |
|     do_eval     |   True   | boolean |  False   |          训练时是否评估          |
|  device_target  | "Ascend" | string  |   True   | 硬件设备,支持 Ascend、GPU 和 CPU |
| data_field_size |    39    |   int   |  False   |          数据集字段数量          |
| data_vocab_size |  184965  |   int   |  False   |          数据集词汇数量          |

### 3.5. 训练输出文件

训练完成后的输出文件如下。

1. 训练输出目录
2. `├─checkpoint`
3. `│ ├─deepfm-xxx.ckpt`
4. `│ ├─deepfm-graph.meta`
5. `├─loss.log`
6. `├─auc.log`
7. `├─deepmf.air`
