-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Graph**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.8.15**

**大小（Size）：174KB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的DCGAN网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述

Tensorflow implementation of Deep Convolutional Generative Adversarial Networks which is a stabilize Generative Adversarial Networks. 

- 论文路径

  https://arxiv.org/abs/1511.06434

- 开源代码路径

  https://github.com/carpedm20/DCGAN-tensorflow

-   适配昇腾 AI 处理器的实现：
    
    https://github.com/Ascend/modelzoo/blob/master/built-in/TensorFlow/Research/graph/DCGAN_ID0686_for_TensorFlow

-   通过Git获取对应commit\_id的代码方法如下：
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   108-hidden, 108-heads
    
-   训练超参（单卡）：
    -   Batch size: 64
    -   epoch: 25
    -   learning_rate: 0.002
    -   beta1: 0.5
    -   train_size: np.inf
    -   input_height:  108
    -   input_width: 108
    -   output_height: 64
    -   output_width: 64
    -   max_to_keep: 1
    -   sample_freq: 200
    -   ckpt_freq: 200
    -   z_dim: 100
    -   z_dist: uniform_signed
    -   G_img_sum:  False


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_performance_1p.sh --help

parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_save_path           the path to save dump/profiling data, default is /home/data
    --data_path		           source data of training
    -h/--help		           show help message
```

相关代码示例:

```
flags.DEFINE_string("precision_mode", "allow_mix_precision", "NPU parameter,allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision")

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
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

1、下载数据集：python download.py mnist celebA

2、目前仅调测了MNIST数据集，下载数据集至./data/mnist路径，用gzip -d命令解压数据集


## 模型训练<a name="section715881518135"></a>
- 下载训练脚本。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练
       
        2. 1单卡训练指令（脚本位于DCGAN_ID0686_for_TensorFlow /test/train_full_1p.sh）
        
            bash test/train_full_1p.sh --data_path=mnist 
    

<h2 id="迁移学习指导.md">高级参考</h2>

## 脚本和事例代码

```
|--main.py       #训练脚本
|	|--......
|--configs		#黑名单算子文件
|   |--ops_info.json
|--test			#训练脚本目录
|	|--train_performance_1p.sh
|	|--train_full_1p.sh
|   |--......
```

## 脚本参数<a name="section6669162441511"></a>

```
	--epoch		      		    Epoch to train,default:25
    --learning_rate             Learning rate of for adam,default:0.0002
    --beta1					    Momentum term of adam,default:0.5
    --train_size		        The size of train images,default:np.inf
    --batch_size				The size of batch images:64
    --input_height				The size of image to use (will be center cropped),default:108
	--input_width				The size of image to use (will be center cropped). If None, same value as input_height [None],default:None
	--output_height				The size of the output images to produce,default:64
	--output_width				The size of the output images to produce. If None, same value as output_height,default:None
	--dataset				    The name of dataset [celebA, mnist, lsun],default:celebA
	--input_fname_pattern		Glob pattern of filename of input images [*],default:*.jpg
	--data_dir					path to datasets [e.g. $HOME/data],default:./data
	--out_dir					Root directory for outputs [e.g. $HOME/out],default:./out
	--out_name					TFolder (under out_root_dir) for all outputs. Generated automatically if left blank [],default:""
	--checkpoint_dir			Folder (under out_root_dir/out_name) to save checkpoints [checkpoint],default:checkpoint
	--sample_dir				Folder (under out_root_dir/out_name) to save samples [samples],default:samples
	--train         			True for training, False for testing [False],default:False
    --crop		         		True for training, False for testing [False], default: False
    --visualize	     			True for visualizing, False for nothing [False], default is False
    --export		 			True for exporting with new batch size, default is False
    --freeze		         	True for exporting with new batch size, default is False
    --max_to_keep		        maximum number of checkpoints to keep,default:1
    --sample_freq               sample every this many iterations,default:200
    --ckpt_freq                 save checkpoint every this many iterations,default:200
    --z_dim                     dimensions of z,default:100
    --z_dist                    normal01' or 'uniform_unsigned' or uniform_signed",default:uniform_signed
    --G_img_sum                 Save generator image summaries in log,default:False
    --precision_mode            NPU parameter,allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision,default:allow_mix_precision
    --over_dump                 Enable OP overflow dump,default:False
    --over_dump_path            Directory name to save the overflow dump files,default:./overflow_dump
    --modify_mixlist            black_list_file,default:./ops_info.json
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。



