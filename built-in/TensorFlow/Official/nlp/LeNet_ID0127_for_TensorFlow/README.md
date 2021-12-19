-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing** 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.4.6**

**大小（Size）：104KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架对图片中手写0~9数字进行识别、分类的训练代码** 

<h2 id="概述.md">概述</h2>

	LeNet是由2019年图灵奖获得者Yann LeCun、Yoshua Bengio于1998年提出(Gradient-based learning applied to document recognition)，它也被认为被认为是最早的卷积神经网络模型。
原始的LeNet是一个5层的卷积神经网络，它主要包括两部分：卷积层，全连接层。本网络的LeNet用于对图片中手写数字进行识别分类。

- 参考论文：

    https://github.com/Jackpopc/aiLearnNotes/blob/master/docs/cv/15-LeNet.md

- 参考实现：

    https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/LeNet.py 

- 适配昇腾 AI 处理器的实现：
    
        
  https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/LeNet_ID0127_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以MNIST训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为28*28
  - 图像输入格式：ubyte

- 测试数据集预处理（以MNIST验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为28*28
  - 图像输入格式：ubyte

- 训练超参

  - Batch size： 64
  - Train epoch: 5
  - Train step: 1000


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(str(args.precision_mode))
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

- 数据集准备
1. 模型训练使用MNIST数据集，数据集请用户自行获取。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
    1. 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

         [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

      单卡训练需要配置指定运行的卡的环境变量：
        export ASCEND_DEVICE_ID=X
        其中：X=0~7

    2. 单卡训练 

        ```
        bash train_full_1p.sh --data_path=../MNIST
        ```
        其中：xxx是数据集的路径，例如, 数据集下载、解压后的路径为"/home/data"，目录结构如下：
           |--data 
           |	|--MINIST
           |        |--t10k-images-idx3-ubyte
           |        |--t10k-labels-idx1-ubyte
           |        |--train-images-idx3-ubyte
           |        |--train-labels-idx1-ubyte
          
         此时，xxx=/home/data/MNIST


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到脚本参数data_path对应目录下。参考代码中的数据集存放路径如下：

     - 训练集： /opt/npu/MNIST
     - 测试集： /opt/npu/MNIST

     数据集也可以放在其它目录，则修改对应的脚本入参data_path即可。


  2. 数据集文件结构，目录参考：

        ```
            |  t10k-images-idx3-ubyte.gz
            |  t10k-labels-idx1-ubyte.gz
            |  train-images-idx3-ubyte.gz
            |  train-labels-idx1-ubyte.gz
        ```


- 加载预训练模型。 
    1. 配置文件参数，修改文件LeNet.py，增加以下参数。
	
        ```
        parser.add_argument('--restore_path', default='/output/ckpt_npu/model.ckpt-100',
        help="""restore path""")            #配置预训练ckpt路径		
        parser.add_argument('--restore_exclude', default=['linear/'],
        help="""restore_exclude""")     #不加载预训练网络中FC层权重
        ```


    2. 模型加载修改，修改文件LeNet.py ，增加以下代码行。
   
        ```
        assert (mode == tf.estimator.ModeKeys.TRAIN)
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=self.args.restore_exclude)
        tf.train.init_from_checkpoint(self.args.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})
        ```

-   模型训练。

    参考“模型训练”中训练步骤。


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── LeNet.py                                  //网络训练与测试代码
├── README.md                                 //代码说明文档
├── LeNet_frozen_graph.py                      //训练模型固化为pb模型代码
├── LeNet_online_inference.py                  //在线推理代码
├── LeNet_preprocess.py                         //在线推理预处理代码
├── requirements.tx                             //训练python依赖列表
├── test
│    ├──train_performance_1p.sh              //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                    //单卡全量训练启动脚本

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path              数据集路径，默认：path/data
--batch_size             每个NPU的batch size，默认：64
--learing_rata           初始学习率，默认：0.001
--steps                  每个epcoh训练步数，默认：1000
--epochs                 训练epcoh数量，默认：5
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  参考脚本的模型存储路径为./output/ckpt_npu。


