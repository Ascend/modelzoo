# XLNET网络迁移

```shell
网络源码路径：https://github.com/zihangdai/xlnet
论文地址：https://arxiv.org/abs/1906.08237
```

### 1、模型概述：

XLNet是一种基于新的广义置换语言建模目标的新的无监督语言表示学习方法。此外，XLNet使用[Transformer-XL](https://arxiv.org/abs/1901.02860) 作为主干模型，对涉及长上下文的语言任务表现出出色的性能。总体而言，XLNet在各种下游语言任务上实现了最先进的（SOTA）结果，包括问题回答、自然语言推理、情绪分析和文档排名。

##### 注：该网络基于20210317主线newest版本迁移。 

### 2、数据集和预训练模型：

数据集名称：[GLUE data](https://gluebenchmark.com/tasks)  

路径：` 10.136.165.4服务器:/turingDataset/datasets/CarPeting_TF_XLNET/glue_data路径`

预训练模型名称： xlnet_cased_L-24_H-1024_A-16

路径：` 10.136.165.4服务器:/turingDataset/datasets/CarPeting_TF_XLNET/xlnet_cased_L-24_H-1024_A-16路径`

### 3、依赖安装: 

`requirements.txt随网络模型归档至gitlab仓。`

```shell
tensorflow==1.15.0
```

### 4、训练详细调测步骤：

#### 训练步骤：

执行：

```shell
全量精度：bash ./startup/run_accuracy_1p.sh
CI性能： bash ./startup/run_performance_1p.sh
```

#### 3、NPU训练结果：

##### `ckpt`文件，loss+perf_npu.txt存放路径：

` 10.136.165.4服务器:/turingDataset/results/CarPeting_TF_XLNET路径`

```shell
INFO:tensorflow:Saving checkpoints for 2 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
I0325 09:29:47.759040 281473812688912 basic_session_run_hooks.py:606] Saving checkpoints for 2 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
INFO:tensorflow:model.ckpt-2 is not in all_model_checkpoint_paths. Manually adding it.
I0325 09:29:59.695219 281473812688912 checkpoint_management.py:95] model.ckpt-2 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Saving checkpoints for 4 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
I0325 09:36:31.501188 281473812688912 basic_session_run_hooks.py:606] Saving checkpoints for 4 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
INFO:tensorflow:model.ckpt-4 is not in all_model_checkpoint_paths. Manually adding it.
I0325 09:36:43.536698 281473812688912 checkpoint_management.py:95] model.ckpt-4 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Saving checkpoints for 6 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
I0325 09:43:15.007579 281473812688912 basic_session_run_hooks.py:606] Saving checkpoints for 6 into /home/cwx1011134/models/model_npu/XLNET_npu/XLNET_org_npu_20210320170034/startup/../ckpt_npu/model.ckpt.
INFO:tensorflow:model.ckpt-6 is not in all_model_checkpoint_paths. Manually adding it.
I0325 09:43:26.997580 281473812688912 checkpoint_management.py:95] model.ckpt-6 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Loss for final step: 6.8431473.
I0325 09:44:16.901361 281473812688912 estimator.py:371] Loss for final step: 6.8431473.


```

#### 4、GPU训练结果：

##### 环境：Tesla V100卡 

##### `ckpt`文件和`graph`文件存放路径：

` 10.136.165.4服务器:/turingDataset/GPU/CarPeting_TF_XLNET/路径`

```shell
精度性能示例：（部分打屏结果如下）
INFO:tensorflow:loss = 11.166861, step = 0
I0325 01:59:36.087652 139798384944960 basic_session_run_hooks.py:262] loss = 11.166861, step = 0
INFO:tensorflow:Saving checkpoints for 2 into ckpt_gpu/model.ckpt.
I0325 01:59:48.834575 139798384944960 basic_session_run_hooks.py:606] Saving checkpoints for 2 into ckpt_gpu/model.ckpt.
INFO:tensorflow:ckpt_gpu/model.ckpt-2 is not in all_model_checkpoint_paths. Manually adding it.
I0325 01:59:53.253901 139798384944960 checkpoint_management.py:95] ckpt_gpu/model.ckpt-2 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Saving checkpoints for 4 into ckpt_gpu/model.ckpt.
I0325 01:59:56.223676 139798384944960 basic_session_run_hooks.py:606] Saving checkpoints for 4 into ckpt_gpu/model.ckpt.
INFO:tensorflow:ckpt_gpu/model.ckpt-4 is not in all_model_checkpoint_paths. Manually adding it.
I0325 02:00:00.471533 139798384944960 checkpoint_management.py:95] ckpt_gpu/model.ckpt-4 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Saving checkpoints for 6 into ckpt_gpu/model.ckpt.
I0325 02:00:03.445174 139798384944960 basic_session_run_hooks.py:606] Saving checkpoints for 6 into ckpt_gpu/model.ckpt.
INFO:tensorflow:ckpt_gpu/model.ckpt-6 is not in all_model_checkpoint_paths. Manually adding it.
I0325 02:00:07.772011 139798384944960 checkpoint_management.py:95] ckpt_gpu/model.ckpt-6 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Loss for final step: 6.5477915.
I0325 02:00:09.900202 139798384944960 estimator.py:371] Loss for final step: 6.5477915.

```

#### 5、GPU/NPU loss收敛趋势：

| step | GPU loss  | NPU loss  |
| :--- | --------- | :-------- |
| 0    | 12.196129 | 5.888282  |
| 100  | 2.5182028 | 2.404872  |
| 200  | 2.2136536 | 1.8116344 |

#### 6、单step耗时 NPU/GPU：

```shell
NPU/GPU=17516.945/54.181

```

