# 训练交付件模板
-   [交付件基本信息](#交付件基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="交付件基本信息.md">交付件基本信息</h2>

_**名称中携带相关基本信息并提供基本属性。此处内容主要用于后续上网作为标签展示。**_

**发布者（Publisher）：_huawei_**

**应用领域（Application Domain）：_可以从以下做选择_**

**_Classification, Object Detection, Segmentation, Image Synthesis, NLP, Speech Synthesis, Speech Recognition, Text To Speech, Audio Synthesis, Machine Translation, Recommendation, Aesthetics Assessment其他（请备注）_**，添加OCR\(推理中，例如CRNN和CTPN Optical Character Recognition，陈海波），Reinforcement Learning。

**版本（Version）：_1.1_**

_版本号规则，2位：_

_第一位，大版本号。0表示C3X，1表示C7X，2及以上预留，根据后续的T/C版本而定_

_第二位，模型自身的升级_

_例如：0.1 -\> 0.2 -\> 1.0 -\> 1.1 -\> … -\> 1.15_

**修改时间（Modified） ：_2020.04.11_**

_**大小（Size）**_**：**_大小，1M以下请直接写xxK，1M以上写xxM；_

如训练后得到的ckpt文件大小。

**框架（Framework）：_第三方框架，TensorFlow、MindSpore、PyTorch等_**_**带框架版本**_

例如：TensorFlow 1.15.0

**模型格式（Model Format）：_ckpt_**

例如：ckpt，_pth等_

**精度（Precision）：_精度 例如 FP32_、FP16、Mixed**

**处理器（Processor）：_昇腾910_**

**应用级别（Categories）：_Bench__mark、Official、Research，tutorial当前先写为__Research_**

_Benchmark__：32__卡训练性能为基线性能1.8__倍（竞品按照线性度测算）_基于昇腾AI处理器，可获得极致性能

_Official__：单卡训练性能为基线性能1.2__倍_  在昇腾AI处理器有良好的精度和性能表现

_Research__：其他_   经典或前沿算法在昇腾AI处理器的实现，供开发者开展研究

**_tutorial_**_：昇腾AI处理器的深度学习快速入门示例_

**描述（Description）：_一句话描述_**

示例：_基于TensorFlow框架的EDVR视频超分网络训练代码_

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

_描述要点（key）：_

-   _对于开源网络，请给出网络简介，并附上参考论文及链接，参考实现链接信息。__注意__不要拷贝论文图片、文字，也不要用自己的语言表达一遍论文的思想。_

    **1. 训练模型代码与推理支持的模型代码同源\(开源地址、分支、Commit ID一致\)。**

    **2. 如果代码来源于开源社区，提供模型开源地址和对应的git分支与Commit ID。**

    EfficientNets是一系列图像分类网络，基于AutoML和Compound Scaling技术搜索得到。相比其他网络，EfficientNets在相似精度的条件下参数量明显更少、或者在相似参数量条件下精度明显更高。EfficientNet-B0是系列网路中最小的基础网络，其他较大尺度网络均基于它缩放生成。本文档描述的EfficientNet-B0是基于Pytorch实现的版本。

    -   参考论文：

        [Tan M, Le Q V. Efficientnet: Rethinking model scaling for convolutional neural networks\[J\]. arXiv preprint arXiv:1905.11946, 2019.](https://arxiv.org/abs/1905.11946)

    -   参考实现：

        ```
        url=https://github.com/lukemelas/EfficientNet-PyTorch.git
        branch=master
        commit_id=3d400a58023086b5c128ecd4b3ea46c129b5988b
        ```


    -   适配昇腾 AI 处理器的实现：
    
        ```
        url=https://gitee.com/ascend/modelzoo.git
        branch=master
        commit_id=9887f0b4ae27f16a1e9f8b0a94dda87b0bf8430a
        code_path=built-in/PyTorch/Official/cv/image_classification/EfficientNet_for_PyTorch
        ```


    通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```
    
    ![](figures/zh-cn_image_0000001093155863.png)
    
    ![](figures/zh-cn_image_0000001093276341.png)

-   _对于自研网络，需要对网络进行较详细的介绍、包括对模型架构的介绍等。_
-   _有分类的提示信息：此处，对我们不同分类的的模型，需要由不同的提示语。_

_Benchmark__（高性能版）：基于昇腾AI__处理器，可获得极致性能_

_Official__（商用版）：在昇腾AI__处理器有良好的精度和性能表现_

_Research__（研究版）：经典或前沿算法在昇腾AI__处理器的实现，供开发者开展研究_

_Tutorials__（新手版）：昇腾AI__处理器的深度学习快速入门示例_

示例：

![](figures/zh-cn_image_0000001093416827.jpg)

## 默认配置<a name="section91661242121611"></a>

_描述要点（key）：_

-   _使用的超参配置__及内部网络实现的优化点。_

    示例：

    ![](figures/zh-cn_image_0000001093276343.png)

-   _数据集信息：对数据集的使用约束或者要求；列举多个可选的数据集，并明确脚本中只是提供了一种参考示例。_

    示例

    ![](figures/zh-cn_image_0000001093560099.png)


## 默认配置示例<a name="section136021153756"></a>

-   网络结构
    -   每个残差分支的最后一个BN采用zero-initialize
    -   卷积采用Kaiming初始化

-   训练数据集预处理（当前代码以ImageNet验证集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

-   测试数据集预处理（当前代码以ImageNet验证集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224（将图像最小边缩放到256，同时保持宽高比，然后在中心裁剪图像）
    -   根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 256
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 0.1
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 90


## 支持特性<a name="section1899153513554"></a>

_描述要点（key）：_

-   _支持的网络和特性__，那些是我们的特性需要定义出来需要讨论：1、分布式训练，2、混合精度，3、数据并行/模型并行，4、小型化_？

示例

![](figures/zh-cn_image_0000001093155857.png)

-   _Features简介_：上面特性的基本概念简介，及如何使用如上特性

## 混合精度训练<a name="section168064817164"></a>

_描述要点（key）：_混合精度训练的基本原理，和Ascend的实现方案简介。

示例：

![](figures/zh-cn_image_0000001093416823.png)

## 开启混合精度<a name="section20779114113713"></a>

_描述要点（key）：如何在该模型下开启混合精度训练_

示例

![](figures/zh-cn_image_0000001093155861.png)

<h2 id="训练环境准备.md">训练环境准备</h2>

_描述要点（key）：_要运行此模型需要具备的硬件要求和软件要求，及参考文档。

示例：

1.  _硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=developer)。需要在硬件设备上安装固件与驱动。_
2.  _宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascend.huawei.com/ascendhub/#/home)获取镜像。_

    _当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。_

    **表 1** _镜像列表_

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-pytorch-arm" target="_blank" rel="noopener noreferrer">ascend-pytorch-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-pytorch-x86" target="_blank" rel="noopener noreferrer">ascend-pytorch-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.1.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://ascend.huawei.com/#/software/cann" target="_blank" rel="noopener noreferrer">20.1</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

_描述要点（key）：_**_如何快速从网上下载脚本并运行训练。我们的方案是否可以做到如此简洁？我们的方案上运行环境是否有差别？比如云环境，实体环境？_**

_是否默认前端环境已经ok？如TensorFlow、依赖等都已经具备。一个容器对应一个网络？还是所有网络兼容同一个容器？_

## 数据集准备<a name="section361114841316"></a>

_请用户自行准备数据集（包括训练集和验证集），例如xx1，xx2，xx3，并上传到train和val文件夹，以xx1为例，数据集格式要求为：_

_（如果是离线预处理请选择此步骤的模板）以xx1数据集为例，用户可以参考如下命令进行数据预处理，用于XX（例如生成H5格式数据集）。_

_**bash scripts/run\_process\_data.sh**_

_（如果是在线预处理请选择此步骤的模板）当前提供的训练脚本中，是以xx1数据集为例，训练过程中进行数据预处理。请用户使用该脚本前，自行修改训练脚本中的数据集和预处理方法。_

示例：

![](figures/zh-cn_image_0000001093416825.png)

## 模型训练<a name="section715881518135"></a>

1.  下载训练脚本。_（例如：单击“立即下载”，并选择合适的下载方式下载源码包。）_
2.  （可选）下载预训练模型。_如果模型需要预训练模型，则需提供。_
3.  上传源码包到服务器并解压。
4.  进入代码目录，编译镜像。_通过Dockerfile编译，代码仓中提供Dockerfile文件。_

    _提供Docker镜像构建命令或脚本。_

    **docker build -t** _\{docker\_image\}_** --build-arg FROM\_IMAGE\_NAME=**\{_base\_image_\}** **.

    示例：

    ![](figures/zh-cn_image_0000001080685154.png)

5.  启动容器实例。

    _可提供启动脚本。_

    **bash scripts/docker\_start.sh **_\[docker\_image__\] \[__data\_dir\] \[__model\_dir\]_

    >![](figures/icon-note.gif) **说明：** 
    >-   _docker\_image_：编译镜像生成的镜像名称
    >-   data\_dir：数据集路径
    >-   model\_dir：训练脚本路径

    示例：

    ![](figures/zh-cn_image_0000001127587081.png)

6.  开始训练。_提供训练环境+配置，让训练跑起来，补充容器化运行的说明。_
    -   单机单卡

        _XXX_

    -   单机八卡

        _XXX_


7.  验证。

    xxx


<h2 id="迁移学习指导.md">迁移学习指导</h2>

_描述要点（key）：_**_提供迁移学习指导和训练：类别 & 数据集的格式说明等_**。_通过用户自定义的数据集能够进行模型训练。_

1.  数据集准备。

    _用户自定义的数据集能够让模型脚本跑起来，提供数据集修改方式。包括数据标注文件格式、目录结构、数据格式转换方法（如有需提供）等。通过修改后的数据集能够直接用于模型训练。_

    1.  获取数据。
    2.  数据目录结构。
    3.  数据标注。_（标注数据格式需要详细说明，提供样例，如脚本中有标注文件样例，可直接写明参考XXX文件。）_

        ![](figures/zh-cn_image_0000001093276349.png)

    4.  （可选）数据转换。_（有则需要提供）_

2.  修改训练脚本。

    _（修改模型配置文件、模型脚本，根据客户实际业务数据做对应模型的修改，以适配）_

    1.  修改配置文件。

        ![](figures/zh-cn_image_0000001093560101.png)

    2.  加载预训练模型。_（预加载模型继续训练或者使用用户的数据集继续训练）_

3.  模型训练。

    _可以参考“模型训练”中训练步骤。（根据实际情况，开源数据集与自定义数据集的训练方法是否一致？）_

4.  模型评估。（根据实际情况）_可以参考“模型训练”中训练步骤。_

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

_描述要点（key）：源码仓目录简介_

示例：

![](figures/zh-cn_image_0000001093560097.png)

## 脚本参数<a name="section6669162441511"></a>

_描述要点（key）：其他参数介绍_

![](figures/zh-cn_image_0000001093276347.png)

![](figures/zh-cn_image_0000001093560095.png)

## 训练过程<a name="section1589455252218"></a>

_描述要点（key）：通过整个训练过程脚本实现的说明介绍里面的原理_

![](figures/zh-cn_image_0000001093416821.png)

## 推理/验证过程<a name="section1465595372416"></a>

_描述要点（key）：通过整个推理过程脚本实现的说明介绍里面的关键信息说明_

![](figures/zh-cn_image_0000001093560093.png)

