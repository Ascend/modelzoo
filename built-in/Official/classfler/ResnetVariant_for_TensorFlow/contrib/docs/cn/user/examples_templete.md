# 示例参考

## 示例说明

新算法的数据集描述：

| Dataset | Default Path | Data Source |
| :--- | :--- | :--: |
| Cifar10 | /cache/datasets/cifar10/ | [下载](https://www.cs.toronto.edu/~kriz/cifar.html) |

新算法的预训练模型描述：

| Algorithm | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| 算法名 | 预训练模型.pth | /cache/models/预训练模型.pth | |

## 示例输入和输出

以下以Prune-EA为例说明

| 阶段 | 选项 | 内容 |
| :--: | :--: | :-- |
| nas | 输入 | 配置文件：compression/prune-ea/prune.yml <br> 预训练模型：/cache/models/resnet20.pth <br> 数据集：/cache/datasets/cifar10 |
| nas | 输出 | 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
| nas | 运行时间估算 | (random_models + num_generation * num_individual) * epochs / GPU数 * 1个epoch的训练时间 |
| fully train | 输入 | 配置文件：compression/prune-ea/prune.yml <br> 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> 数据集：/cache/datasets/cifar10 |
| fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
| fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |
