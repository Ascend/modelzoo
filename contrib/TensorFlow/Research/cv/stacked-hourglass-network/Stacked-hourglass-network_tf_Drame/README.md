# Stacked Hourglass Networks for Human Pose Estimation 人体姿态估计

## 1. 下载连接

论文源码地址：https://github.com/wbenbihi/hourglasstensorlfow （Tensorflow）

MPII数据集下载地址：http://human-pose.mpi-inf.mpg.de/#download


## 2. 结果分析

论文精度：90.9%

在NPU上实现的精度：91.180 %

## 3. 文件说明

images：存放MPII数据集图片

code：存放训练代码

code/config.cfg: 超参数和文件地址
- training_txt_file：dataset.txt文件的地址
- img_directory：MPII图片数据集地址
- log_dir_train 和 log_dir_test：训练和测试产生的event文件地址
- 模型相关参数

			nStack				: 堆叠模型的数量
			nFeat				: 卷积通道数量
			nLow				: 每个模块下采样的数量
			outputDim			: 输出维度
			batch_size			: 批处理大小
			drop_rate			: 每层神经元失活率
			lear_rate			: 厨师学习率
			decay				: 指数衰减学习率，在0到1之间
			decay_step			: 应用衰减的步数			
			dataset			        : 数据集
			training			: (bool) 训练时为True，验证时为False
			w_summary			: (bool) 是否保存权重
			tiny				: (bool) 是否使用小的漏斗模型
			attention			: (bool) 是否使用Attention机制
			modif				: (bool) 测试网络修改的地方，一般不使用
			name				: checkpoint的文件名前缀

## 4. 从头训练

```
cd ..../code
python3.7 train_launcher.py
```
## 5. 加载checkpoint训练

将下载的Checkpoint文件放在code目录下

```
cd ..../code
python3.7 load.py
```

##  6. License规则(tensorflow 迁移场景)
```
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

