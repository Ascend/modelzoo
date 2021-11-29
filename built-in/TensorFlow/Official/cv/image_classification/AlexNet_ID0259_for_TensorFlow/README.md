- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.9.22**

**大小（Size）：174KB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的图片分类识别网络训练代码**

<h2 id="概述.md">概述</h2>
-     AlexNet是2012年Hinton和他的学生Alex Krizhevsky设计的 深度卷积神经网络模型，可以算是LeNet的一种更深更宽的版本 学习网络。 AlexNet主要使用到的新技术点如下： 

（1）成功使用ReLU作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题。

（2）训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。

（3）在CNN中使用重叠的最大池化。

（4）提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

（5）使用CUDA加速深度卷积网络的训练，利用GPU强大的并行计算能力，处理神经网络训练时大量的矩阵运算。

（6）数据增强，随机地从256*256的原始图像中截取224*224大小的区域（以及水平翻转的镜像），相当于增加了2*(256-224)^2=2048倍的数据量。



- 参考论文：

    ```
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    ```

- 参考实现：
    ```
    https://github.com/demul/AlexNet
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/AlexNet_ID0259_for_TensorFlow
    ```


- 通过Git获取对应commit_id的代码方法如下:

    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置 <a name="section91661242121611"></a>
-   网络结构
    5层卷积层和3层全连接层 
    
-   训练超参（单卡）：
    -   Batch size: 100
    -   epoch: 35
    -   learning_rate: 0.001
    -   momentum: 0.9
    -   LRN_depth: 5
    -   LRN_bias: 2
    -   lRN_alpha: 0.0001
    -   LRN_beta: 0.75
    -   keep_prob: 0.5


## 支持特性 <a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练 <a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度 <a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

相关代码示例:



```
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

<h2 id="训练环境准备.md">训练环境准备</h2>
1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>
## 数据集准备 <a name="section361114841316"></a>

1. 自行下载“Catvs Dog” Datasets，25000张中有24000张是Training set，1000张是Test set。


## 模型训练 <a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包

- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练
        
   
        ```        
        bash test/train_full_1p.sh --data_path=/home/Dog_Cats
        ```

​   
<h2 id="高级参考.md">高级参考</h2>
## 脚本和示例代码 <a name="section08421615141513"></a>

    |--main.py                 #训练脚本入口
    |--train.py		           #执行训练主脚本
    |--model.py                #构建模型脚本
    |--test			           #训练脚本目录 
    |	|--train_performance_1p.sh
    |	|--train_full_1p.sh
    |--data_loader.py          #数据集加载脚本
    |--data_preprocessor.py    #数据集预处理脚本



## 脚本参数 <a name="section6669162441511"></a>

```
    --epoch                       Epoch to train,default:35
    --learning_rate             Learning rate of for adam,default:0.001
    --data_path                  dataset path
    --batch_size                 batch size，default：100
```

## 训练过程 <a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/../model，包括训练的log以及checkpoints文件。loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。



