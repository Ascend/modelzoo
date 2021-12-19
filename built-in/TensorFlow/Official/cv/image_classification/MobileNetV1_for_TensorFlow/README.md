-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.7.21**

**大小（Size）：51.7M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MobileNetV1图像分类网络训练代码**



<h2 id="概述.md">概述</h2>

MobileNetv1专注于移动端或者嵌入式设备中的轻量级CNN网络。是将传统的卷积操作改为两层的卷积操作。它基于流线型架构，该架构使用深度可分离卷积来构建轻量级深度神经网络。MobileNets 在广泛的应用程序和用例中的有效性，包括对象检测、细粒度分类、人脸属性和大规模地理定位。

-   参考论文：

	https://arxiv.org/abs/1704.04861


-   参考实现：

    [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim)
        

-   适配昇腾 AI 处理器的实现：
    
    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/MobileNetV1_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
   
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

    

## 默认配置<a name="section91661242121611"></a>


-   训练超参：
    -   Batch size: 256
    -   Momentum: 0.9
    -   LR scheduler: cosine annealing
    -   Learning rate\(LR\): 0.4
    -   Weight decay: 0.00004
    -   Label smoothing: 0.1
    -   Train epoch: 300
    -	Warmup_epoch: 5
    -	Optimizer: MomentumOptimizer
    -	Moving average decay=0.9999



## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，具体的参数在estimator_impl.py脚本中（脚本位于MobileNetV2_for_TensorFlow目录下）。设置precision_mode参数的脚本参考如下。


```
run_config = NPURunConfig(
             hcom_parallel=True,
             precision_mode="allow_mix_precision",
             enable_data_pre_proc=True,
             save_checkpoints_steps=self.env.calc_steps_per_epoch(),
             session_config=self.estimator_config,
             model_dir=logdir,
             iterations_per_loop=config['iterations_per_loop'],
             keep_checkpoint_max=5
        )
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

## 数据集准备<a name="section361114841316"></a>

1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_performance_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --dataset_dir=/npu/traindata/imagemat_TF
     ```

  2. 启动训练。

     启动单卡性能训练 （脚本为MobileNetV1_for_TensorFlow/test/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh
     ```
	 
	 启动单卡精度训练 （脚本为MobileNetV1_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh
     ```
- 8卡训练

  1. 配置训练参数。

     首先在脚本test/train_full_8p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --dataset_dir=/npu/traindata/imagemat_TF
     ```

  2. 启动训练。

     启动8卡性能训练 （脚本为MobileNetV1_for_TensorFlow/test/train_performance_8p.sh） 

     ```
     bash train_performance_8p.sh
     ```
	 
     启动8卡精度训练 （脚本为MobileNetV1_for_TensorFlow/test/train_full_8p.sh） 

     ```
     bash train_full_8p.sh
     ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── train.py                                          //网络训练与测试代码
├── env.py                                            //超参配置
├── README.md                                         //说明文档
├── logger.py
├── eval_image_classifier_mobilenet.py                //测试脚本
├── dataloader
│    ├──data_provider.py                             //数据加载入口脚本 
├── estimator_impl.py
```

## 脚本参数<a name="section6669162441511"></a>

```
--dataset_dir              数据集路径，默认：/opt/npu/slimImagenet
--max_train_steps          最大的训练step 数， 默认：None
--iterations_per_loop      NPU运行时，device端下沉次数，默认：None
--model_name               网络模型，默认：mobilenet_v2_140
--moving_average_decay     滑动平均的衰减系数， 默认：None
--label_smoothing          label smooth 系数， 默认：0.1
--preprocessing_name       预处理方法， 默认：inception_v2
--weight_decay             正则化系数，默认：0
--batch_size               每个NPU的batch size， 默认：256
--learning_rate_decay_type 学习率衰减的策略， 默认：fixed
--learning_rate            学习率， 默认：0.1
--optimizer                优化器， 默认：sgd
--momentum                 动量， 默认：0.9 
--warmup_epochs            学习率线性warmup 的epoch数， 默认：5
--max_epoch                训练epoch次数，默认：300
```


## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动性能或者精度训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为result/8p/*。

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本的参数checkpoint_path可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印验证集的top1 accuracy，，打印在eval_*.log中
