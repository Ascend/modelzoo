-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.7.20**

**大小（Size）：90.1M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ResNet200图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

1. ResNet在ImageNet竞赛中是一个比较好的分类问题网络。通过增加直接连接通道来保护信息的完整性，解决了信息丢失、梯度消失、梯度爆炸等问题。这个网络也可以训练。ResNet有不同的网络层，常用的有18层、34层、50层、101层、152层。


- 参考论文：

    [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
	
- 参考实现：

  https://github.com/tensorflow/models/tree/r2.1.0/official/r1/resnet
  
- 适配昇腾 AI 处理器的实现：
          
  https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/ResNet200_for_TensorFlow
  
- 通过Git获取对应commit\_id的代码方法如下： 
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```
## 默认配置<a name="section91661242121611"></a>
- 训练数据集预处理

  请参考"概述"->"参考实现"。

- 训练超参

    - resnet_size: 200
    - batch_size: 512
    - num_gpus: 1 
    - cosine_lr: True 
    - dtype: fp16 
    - label_smoothing: 0.1 
    - loss_scale: 512 
    - max_train_steps: 2000 
    - train_epochs: 1
    - eval_only: False 
    - hooks: ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook 
	- data_format: "channels_last" 

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  run_config = NPURunConfig(        
  		model_dir=flags_obj.model_dir,        
  		session_config=session_config,        
  		keep_checkpoint_max=5,        
  		save_checkpoints_steps=5000,        
  		enable_data_pre_proc=True,        
  		iterations_per_loop=iterations_per_loop,        			
  		log_step_count_steps=iterations_per_loop,        
  		precision_mode='allow_mix_precision',        
  		hcom_parallel=True      
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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.1</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
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
      --data_dir=/npu/traindata/imagemat_TF
     ```

  2. 启动训练。

     启动单卡性能训练 （脚本为ResNet200_for_TensorFlow/test/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh
     ```

	 启动单卡精度训练 （脚本为ResNet200_for_TensorFlow/test/train_full_1p.sh） 
	
	 ```
	 bash train_full_1p.sh
	 ```
- 8卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_8p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_dir=/npu/traindata/imagemat_TF
     ```

  2. 启动训练。

     启动8卡性能训练 （脚本为ResNet200_for_TensorFlow/test/train_performance_8p.sh） 

     ```
     bash train_performance_8p.sh
     ```
		
     启动8卡精度训练 （脚本为ResNet200_for_TensorFlow/test/train_full_8p.sh） 

     ```
     bash train_full_8p.sh
     ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```                                
├── README.md  
├── r1                   
│	├── resnet
│		├──cifar10_download_and_extract.py
│		├──cifar10_main.py
│		├──cifar10_test.py
│		├──estimator_benchmark.py
│		├──frozen_graph.py
│		├──imagenet_main.py
│		├──imagenet_preprocessing.py
│		├──imagenet_test.py
│		├──infer_from_pb.py
│		├──resnet_model.py
│		├──resnet_run_loop.py
│		├──run_imagenet.sh           
├── test
│    ├──train_performance_1p.sh              
│    ├──train_performance_8p.sh             
│    ├──train_full_1p.sh                     
│    ├──train_full_8p.sh                     
│    ├──env.sh                                  
```

## 脚本参数<a name="section6669162441511"></a>

```
--resnet_size		      		 # resnet的系列的模型类别 
--batch_size			 		 # batchsize
--num_gpus						 # NPU设备数量
--cosine_lr						 # 是否使用余弦的学习率策略
--dtype=						 # fp16 为设置为半精度训练，可与mixed精度配合使用
--label_smoothing				 # label smooth的参数
--loss_scale					 # 用于混合精度训练的loss scale
--max_train_steps				 # 最大训练步数
--train_epochs					 # 总训练epoch数
--eval_only						 # 是否只做评估，训练时需要设置为False
--hooks                          # hook对象
--data_dir						 # 数据集路径
--data_format	  				 # channel表示在前
--model_dir                      # 模型、log等保存路径
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动性能或者精度训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为test/output/*。

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印验证集的top1 accuracy和top5 accuracy，打印在train_*.log中


