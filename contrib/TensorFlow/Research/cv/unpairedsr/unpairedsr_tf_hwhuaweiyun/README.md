### 项目概述

本项目复现 [Unpaied-SR](https://arxiv.org/abs/1807.11458) 模型，项目代码基于 TensorFlow 框架，可运行于 Ascend 910，精度比对结果如下表：

|      | 原论文 | Ascend 910 | NVIDIA TITAN Xp |
| ---- | :----: | :--------: | :-------------: |
| PSNR | 19.30  |   19.87    |      19.90      |

###  目录结构

```
unpairedsr
├── LICENSE
├── README.md
├── data                # 数据集
   ├── dev
│   │   ├── SRtrainset_2_1.npy
│   │   └── SRtrainset_2_2.npy
│   ├── test
│   │   └── LS3D_6000.npy
│   └── train
│       ├── celea_60000_SFD_1.npy
│       ├── ...
│       └── vggcrop_train_lp10_5.npy
├── low_high_model.py   # 模型代码
├── main.py             # 训练与测试代码
└── output              # 训练结果
    ├── checkpoint
    ├── ...
    └── model-0.meta
```

### 数据集说明

data 目录下需要包含 train、test、dev 三个子目录，分别对应训练集、测试集、开发集。每个子目录下是若干个 npy 文件，包含经过预处理的数据，数据以字典格式存储，包含两个键：

* sample：低分辨率图像
* label：高分辨率图像

[数据文件](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WrkTg8PzzdAMSqfXzuGlEoPHJVkuOxg0zuDH1271OXQZDVQPHn/B4tObo9vCz5MUzmD1/1iG3/n44iZpfFQganJk7y0NZtrN+PeK7/unvLf0fr5hZR0VSqgFbG7OJI9bB44cxY9fy6vGJnWP4zaQG26E5nqBxMz7ljFWUs+PF9GW1RvPZCHHyadE1j2nzwDFmvCg+D3+LY8V1kqXG14X1gabPhQAMStwPlBsrUeDXSzuDtdKV8bsZO17t/7lDHYAElMxDJYp20fkuKgNfxZKoDcnicvifl9xMHkQLOKyAq9HcVVgSUdO/+wLFh6rwDtct0XL3tQCkwbEQuU0cJ9kkxRL0SsZXDwUyb9Iu88tYr+GnkyjDAIgpyuiZnCgLZNEHXn8xotsdJBlJiIeT66ThrUyW2Xv5w75ibPtgEg+trzVT2PbN4NIL9yb7aMe+l/wEeTI4rJDlWjW5T0rN00xS9Xy+4H7Ao6CWaBZPFilWClQQDj0JbIQgSaqS3RKeq0lkbtrUE0PLClyd2eQNJJ8isEr4rwakODtB/uNb3hNY2QrMjNhO0nzt81ksLLK4ETj)（提取码：123456）

### 模型文件

训练过程中，模型会在开发集上进行测试，在开发集上取得最佳性能的模型文件会保存在 output 目录下。

[预训练模型文件](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WrkTg8PzzdAMSqfXzuGlEoPHJVkuOxg0zuDH1271OXQZDVQPHn/B4tObo9vCz5MUzmD1/1iG3/n44iZpfFQganJk7y0NZtrN+PeK7/unvLf0fr5hZR0VSqgFbG7OJI9bB44cxY9fy6vGJnWP4zaQG26E5nqBxMz7ljFWUs+PF9GW1RvPZCHHyadE1j2nzwDFmvCg+D3+LY8V1kqXG14X1gabPhQAMStwPlBsrUeDXSx50wxYBSB3ZRXmnbxbYa6WDvftD1a0FSAq5dM7cg1Jjs/e/V8lFeF8kl3gLFVYv0P8egIODl/3ZfFiIl1/USt/7SnGv2ZJ0m9JEnvafY/tVM9NN1GjetqzNUkajkQ8zV6nylIa48/Uxw8PCKA7Y2lsQkBZ8i1KEMZ0E0I6dZABcHko1cGByNmP77Qs71sm/DjjTo0syzHQBs2hvjOsCv1G/rRNeuk89WW5fIUtNh2XuANuScLSegSxxFLw7f93Y3kYWH/Lm6B+6oNIhIxdBBk3ZDAetkuM/mdr8HJe1doTrK4goyMIeZlX4RMHYR++hqAJ+hqHuoNXMTVUcS7UofV92Ter1K5WkYjySQ09k6WE7g==)（提取码：123456）

### 参数说明

```
main.py [-h] [--train] [--test] [--max_epoch MAX_EPOCH]
             [--batch_size BATCH_SIZE] [--print_interval PRINT_INTERVAL]
             [--n_generator N_GENERATOR] [--alpha ALPHA] [--beta BETA]
             [--learning_rate LEARNING_RATE] [--data_dir DATA_DIR]
             [--model_dir MODEL_DIR]
--train：模型训练
--test：模型测试
--max_epoch：最大训练轮数，默认值 50
--batch_size：批量大小，默认值 32
--print_interval：每print_interval个训练步数进行一次验证，默认值 300
--n_generator：判别器与生成器的更新次数比例，默认值 5
--alpha：mse loss的权重，默认值 1.0
--beta：generator loss的权重，默认值 0.05
--learning_rate：学习率，默认值 0.0001
--data_dir：数据目录，默认值 './data'
--model_dir：模型文件目录，训练阶段，该目录用于保存最佳模型，测试阶段，该目录下的模型文件用于测试，默认值 './output'
```

### 训练

* 下载数据集，放置在 data 目录下
* 执行训练命令：`python main.py --train`

### 测试

* 下载数据集，放置在 data 目录下
* 下载预训练模型，放置在 output 目录下
* 执行测试命令：`python main.py --test`

