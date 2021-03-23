-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.2**

**修改时间（Modified） ：2020.10.14**

**大小（Size）：221M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ResNeXt-50图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

ResNeXt网络在ResNet基础上进行了优化，同时采用Vgg/ResNet堆叠的思想和Inception的split-transform-merge思想，把单路卷积转变成了多个支路的多个卷积。ResNeXt结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量。ResNeXt有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。Ascend本次提供的是50-layer的ResNeXt-50网络。

 -   参考论文：

        [Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) 

 -   参考实现：
        
 
 -   适配昇腾 AI 处理器的实现：
    
        ```
        https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/ResNext50_for_TensorFlow
        branch=master
        commit_id=be78dd41dd3744a2b21c13a62eba829d59b111f2
        ```


    通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ImageNet-Train训练集为例，仅作为用户参考示例）：

  图像的输入尺寸为224*224

  图像输入格式：TFRecord

  数据集大小：1281167

- 测试数据集预处理（以ImageNet-Val验证集为例，仅作为用户参考示例）：

  图像的输入尺寸为224*224

  图像输入格式：TFRecord

  验证集大小：50000

- 训练超参（8卡）：

  Batch size: 32

  Momentum: 0.9

  loss_scale：1024

  LR scheduler: cosine

  Learning rate(LR): 0.1

  learning_rate_end: 0.000001

  warmup_epochs: 5

  train epoch: 120

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。 当前昇腾910 AI处理器仅支持float32到float16的精度调整。使用自动混合精度功能后，推荐开启Loss Scaling，从而补偿降低精度带来的精度损失。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。 
    
```
    run_config = NPURunConfig( 
                hcom_parallel=True, 
                precision_mode='allow_mix_precision', 
                enable_data_pre_proc=True, 
                save_checkpoints_steps=50000, 
                session_config=session_config, 
                model_dir = self.config['model_dir'], 
                iterations_per_loop=self.config['iterations_per_loop'], 
                keep_checkpoint_max=5 
            )
```

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  _硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。_
2.  _宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。_

    _当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

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

- 准备数据集

    1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。。
    2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。
    3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练

    1. 修改 `ResNext50_for_TensorFlow / bin ` 中  `npu_set_env_1p.sh` 文件,根据hccl_1p.json文件实际路径进行配置，如下：

        ```
        export RANK_TABLE_FILE=/network/ResNext50_for_TensorFlow/hccl_1p.json
        ```

    2. 修改 `ResNext50_for_TensorFlow / testscript` 中 `Resnext50_1p_host.sh` 文件，根据文件 `npu_set_env_1p.sh` 的实际路径进行配

       置，如下：

        ```
        source /network/ResNext50_for_TensorFlow/bin/npu_set_env_1p.sh $RANK_ID $RANK_SIZE
        ```

    3. 根据文件 `res50.py` 的实际路径进行配置修改

        ```
       python3.7 ResNext50_for_TensorFlow/code/resnext50_train/mains/res50.py --config_file=res50_32bs_1p_host --max_train_steps=10000 --  iterations_per_loop=1000   --debug=True  --eval=False  --model_dir=./  2>&1 | tee train_$RANK_ID.log &
        ```

    4. 修改`ModelZoo_ResNext50_TF_MTI\code\resnext50_train\configs` 中`res50_32bs_1p_host`文件:

       4.1 配置参数data_url为数据集的路径地址，确保修改为用户数据集所在路径，如下：

        ```
        'data_url':  'file:///ModelZoo_ResNext50_TF_MTI/data/resnext50/imagenet_TF',
        ```

       4.2 配置checkpoint文件的路径，确保修改后路径正确，如下

         ```
        'ckpt_dir': 'ModelZoo_ResNext50_TF_MTI/d_solution/ckpt0',
         ```

    5. 执行训练脚本:

       5.1 进行训练时，需要使用 `res50_32bs_1p_host.py` 脚本参数（脚本位于`ModelZoo_ResNext50_TF_MTI/code / resnext50_train / configs / res50_32bs_1p_host.py`）， `mode` 默认设置为         `train`。

            ```
            'mode':'train',                                         # "train","evaluate"

            ```
       5.2 进行验证时，需要修改 `res50_32bs_1p_host.py` 脚本参数（脚本位于`ModelZoo_ResNext50_TF_MTI/code / resnext50_train / configs / res50_32bs_1p_host.py`），将 `mode` 设置为 `evaluate`。

            ```
            'mode':'evaluate',                                         # "train","evaluate"
            ```
        
       5.3 单卡训练指令（脚本位于`ModelZoo_ResNext50_TF_MTI/testscript/Resnext50_1p_host.sh`）
            
            ```
            bash  Resnext50_1p_host.sh
            ```

- 8卡训练

    1. 修改 `ResNext50_for_TensorFlow / bin ` 中 `npu_set_env.sh`文件,根据hccl.json文件实际路径进行配置，如下：

        ```
        export RANK_TABLE_FILE=/network/ResNext50_for_TensorFlow/hccl.json
        ```

    2. 修改 `ResNext50_for_TensorFlow / testscript` 中 `Resnext50_8p_host.sh` 文件，根据文件 `npu_set_env.sh` 的实际路径进行配

       置，如下：

        ```
        source /network/ResNext50_for_TensorFlow/bin/npu_set_env.sh $RANK_ID $RANK_SIZE
        ```

    3. 根据文件 `res50.py` 的实际路径进行配置修改

        ```
       python3.7 ResNext50_for_TensorFlow/code/resnext50_train/mains/res50.py --config_file=res50_32bs_8p_host --max_train_steps=10000 --  iterations_per_loop=1000   --debug=True  --eval=False  --model_dir=./  2>&1 | tee train_$RANK_ID.log &
        ```

    4. 修改`ModelZoo_ResNext50_TF_MTI\code\resnext50_train\configs` 中`res50_32bs_8p_host`文件:

       4.1 配置参数data_url为数据集的路径地址，确保修改为用户数据集所在路径，如下：

        ```
        'data_url':  'file:///ModelZoo_ResNext50_TF_MTI/data/resnext50/imagenet_TF',
        ```

       4.2 配置checkpoint文件的路径，确保修改后路径正确，如下

         ```
        'ckpt_dir': 'ModelZoo_ResNext50_TF_MTI/d_solution/ckpt0',
         ```

    5. 执行训练脚本:

       5.1 进行训练时，需要使用 `res50_32bs_8p_host.py` 脚本参数（脚本位于`ModelZoo_ResNext50_TF_MTI/code / resnext50_train / configs / res50_32bs_8p_host.py`）， `mode` 默认设置为         `train`。

            ```
            'mode':'train',                                         # "train","evaluate"
            ```
       
       5.2 进行验证时，需要修改 `res50_32bs_8p_host.py` 脚本参数（脚本位于`ModelZoo_ResNext50_TF_MTI/code / resnext50_train / configs / res50_32bs_8p_host.py`），将 `mode` 设置为 `evaluate`。

            ```
            'mode':'evaluate',                                         # "train","evaluate"
            ```
        
       5.3 8P训练指令（脚本位于`ModelZoo_ResNext50_TF_MTI/testscript/Resnext50_8p_host.sh`）
            
            ```
            bash Resnext50_8p_host.sh
            ```

- 验证。

    1. 通过“模型训练”中的测试指令启动8P测试。

        1.1 进行验证之前，需要修改 `res50_32bs_8p_host.py` 脚本参数（脚本位于`ModelZoo_ResNext50_TF_MTI/code / resnext50_train / configs / res50_32bs_8p_host.py`），将 `mode` 设置为 `evaluate`。

            ```
            'mode':'evaluate',                                         # "train","evaluate"
            ```
        1.2 8P测试指令（脚本位于`ModelZoo_ResNext50_TF_MTI/testscript/Resnext50_8p_host.sh`）
            
            ```
            bash Resnext50_8p_host.sh    

    2. 在验证执行完成后，验证结果会输出到 ModelZoo_ResNext50_TF_MTI/result/cloud-localhost--0/0/resnet50_train/results/res50_32bs_8p目录中。 测试结束后会打印

验证集的top1 accuracy和top5 accuracy。


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。
    数据集要求如下：
    1.  数据集要求如下： 数据准备。 如果要使用自己的数据集，需要将数据集放到如下目录：

        - 训练集： data\resnext50

        - 测试集： data\resnext50

        类别数可以通过训练参数中的num_classes来设置。

    2.  准确标注类别标签的数据集。

    3.  数据集每个类别所占比例大致相同。

    4.  数据集文件结构，请用户自行参照tfrecord脚本生成train/eval使用的TFRecord文件，包含训练集和验证集两部分，目录参考：
        
       
```
        |--|imagenet_tfrecord
        |   train-00000-of-01024
        |   train-00001-of-01024
        |   train-00002-of-01024
        |   ...
        |   validation-00000-of-00128
        |   validation-00000-of-00128
        |   ...
```
    5.  设置合理的数据集预处理方法（裁剪大小、随机翻转、标准化）。
        
```
        def parse_and_preprocess_image_record(config, record, height, width,brightness, contrast, saturation, hue,
                                              distort, nsummary=10, increased_aug=False, random_search_aug=False):
                with tf.name_scope('preprocess_train'):
                    image = crop_and_resize_image(config, record, height, width, distort)   #解码，80%中心抠图并且Resize[224 224]
                    if distort:
                        image = tf.image.random_flip_left_right(image)            #随机左右翻转
                        image = tf.clip_by_value(image, 0., 255.)                     #归一化
                image = normalize(image)                  #减均值[121.0, 115.0, 100.0]，除方差[70.0, 68.0, 71.0]
                image = tf.cast(image, tf.float16)
                return image
```

- 修改训练脚本。

    1.  修改配置文件。

        模型分类类别修改。 
        
        1.1 使用自有数据集进行分类，如需将分类类别修改为10。 修改code/resnext50_train/models/resnet50/resnet.py文件，将units=1001设置为units=10。 
        
            
```
            axes = [1,2] 
            x = tf.reduce_mean( x, axes, keepdims=True ) 
            x = tf.identity(x, 'final_reduce_mean') 
            x = tf.reshape( x, [-1, 2048] ) 
            x = tf.layers.dense(inputs=x, units=1001,kernel_initializer= tf.variance_scaling_initializer() ) 
            . . . 
            x = tf.layers.dense(inputs=x, units=1001,kernel_initializer=tf.random_normal_initializer(stddev=0.01)) 
```

        1.2 修改code/resnext50_train/models/resnet50/res50_model.py文件，将depth=1001设置为depth=10。
            `labels_one_hot = tf.one_hot(labels, depth=1001) `
        1.3 修改code/resnext50_train/configs/res50_32bs_1p_host.py文件，将num_classes=1001设置为num_classes=10。 
        
```
            'model_name': 'resnet50', 
            'num_classes': 1001,
```


    2.  加载预训练模型。

        2.1 配置文件增加参数，修改code/resnext50_train/configs/res50_32bs_1p_host.py文件（具体配置文件名称，用户根据自己实际名称设置），增加以下参数。                     
        
        `'restore_path': '/code/ckpt0/model.ckpt-601000', `
        2.2 用户根据预训练的实际ckpt进行配置 
        `'restore_exclude': ['fp32_vars/dense'], `
        2.3 不加载预训练网络中FC层权重 模型加载修改，修改code/resnext50_train/models/resnet50/res50_model.py文件，增加以下代码行。 
        
```
        #restore ckpt for finetune， 
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=self.config.get('restore_exclude'))         
        tf.train.init_from_checkpoint(self.config.get('restore_path'),{v.name.split(':')[0]: v for v in variables_to_restore})
```
 

- 模型训练。

    _可以参考“模型训练”中训练步骤。_

- 模型评估。

    _可以参考“模型训练”中训练步骤。_

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>


```
    ├── code 
    │ ├──resnext50_train //训练脚本代码
    │ │ ├──configs //参数配置脚本
    │ │ │ ├── res50_32bs_1p_host.py //单卡参数配置脚本
    │ │ │ ├── res50_32bs_8p_host.py //8卡参数配置脚本
    ├── bin
    │ ├──npu_set_env.sh //8卡环境变量配置脚本 
    │ ├──npu_set_env_1p.sh //单卡环境变量配置脚本
    ├── testscript 
    │ ├──Resnext50_1p_host.sh //物理机场景：1P执行脚本 
    │ ├──Resnext50_8p_host.sh //物理机场景：8P执行脚本 
    ├── hccl.json    //8卡device配置脚本
    ├── hccl_1p.json //单卡device配置脚本
    ├── frozen_graph.py //转PB模型脚本
    ├── requirements.txt //相关依赖说明脚本
```

## 脚本参数<a name="section6669162441511"></a>

    --rank_size 使用NPU卡数量，默认：单P 配置1，8P 配置8 
    
    --mode 运行模式，默认train；可选：train，evaluate 

    --max_train_steps 训练次数，单P 默认：10000 

    --iterations_per_loop NPU运行时，device端下沉次数，默认：1000 

    --eval 训练结束后，是否启动验证流程。默认：单P False，8P True 

    --num_epochs 训练epoch次数， 默认：单P None，8P 120 

    --data_url 数据集路径，默认：data/resnext50/imagenet_TF 

    --ckpt_dir 验证时checkpoint文件地址 默认：/d_solution/ckpt0 

    --lr_decay_mode 学习率方式，默认：cosine

    --learning_rate_maximum 初始学习率，默认：0.1 

    --learning_rate_end 结束学习率：默认：0.000001 

    --batch_size 每个NPU的batch size，默认：32 

    --warmup_epochs 初始warmup训练epoch数，默认：5 

    --momentum 动量，默认：0.9

    说明：当前默认模式为train，iterations_per_loop配置1000，每训练1000step会在日志中打印出 loss，FPS，和lr。在8P训练脚本中，配置了eval=True，当训练结束后，会进行验证流程，输出精度验证结果。

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持1，8P网络训练。训练执行过程中，会输出step，FPS，loss到如下路径。 /result/cloud-localhost--0/0/resnet50_train/results/res50_32bs_1p 

```
    Step Epoch Speed Loss FinLoss LR 
    step: 1000 epoch: 0.0 FPS: 180.0 loss: 6.797 total_loss: 8.258 lr:0.1000000 
    step: 2000 epoch: 0.0 FPS: 741.5 loss: 6.766 total_loss: 7.980 lr:0.1000000 
    step: 3000 epoch: 0.1 FPS: 741.5 loss: 6.707 total_loss: 7.730 lr:0.1000000 
    step: 4000 epoch: 0.1 FPS: 741.3 loss: 6.496 total_loss: 7.367 lr:0.1000000 
    step: 5000 epoch: 0.1 FPS: 741.4 loss: 6.410 total_loss: 7.164 lr:0.1000000 
    step: 6000 epoch: 0.1 FPS: 741.4 loss: 6.148 total_loss: 6.812 lr:0.1000000 
    step: 7000 epoch: 0.2 FPS: 741.2 loss: 5.910 total_loss: 6.504 lr:0.1000000 
    step: 8000 epoch: 0.2 FPS: 741.5 loss: 6.117 total_loss: 6.656 lr:0.1000000
    
```

## 推理/验证过程<a name="section1465595372416"></a>

 通过“模型训练”中的测试指令启动8P测试。在120 epoch训练执行完成后，脚本会自动执行验证流程。验证结果会输出到 /result/cloud-localhost--0/0/resnet50_train/results/res50_32bs_8p目录中。 

测试结束后会打印验证集的top1 accuracy和top5 accuracy，如下所示。
 
```
Evaluating 
Validation dataset size: 50000 
step   epoch top1   top5 loss checkpoint_time(UTC) 
226000 46.0 61.611 84.63 2.55 2020-06-21 01:33:11 
339000 68.0 67.316 88.64 2.42 2020-06-21 01:33:12 
452000 91.0 72.931 91.71 2.26 2020-06-21 01:33:12 
565000 113.0 77.869 93.99 2.17 2020-06-21 01:33:11 
601000 121.0 78.197 94.14 2.18 2020-06-21 01:33:11 
Finished evaluation
```
