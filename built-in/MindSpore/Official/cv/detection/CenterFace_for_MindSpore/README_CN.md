# 目录

- [目录](#目录)
- [CenterFace 描述](#centerface描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本描述](#脚本描述)
  - [脚本和样例代码](#脚本和样例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [训练](#训练)
  - [测试过程](#测试过程)
    - [测试](#测试)
  - [评估过程](#评估过程)
    - [评估](#评估)
  - [转换过程](#转换过程)
    - [转换](#转换)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [评估性能](#评估性能)
    - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo 主页](#modelzoo主页)

# [CenterFace 描述](#目录)

CenterFace 是一个 anchor-free 的轻量级人脸检测网络，主要用于边缘设备，我们支持 A910 的训练和评估。

无约束环境下的人脸检测和对齐通常部署在内存有限、计算能力低的边缘设备上。CenterFace 提出一种实时快速、高精度的同时预测人脸框和关键点的一阶段方法。

[论文](https://arxiv.org/ftp/arxiv/papers/1911/1911.03599.pdf): CenterFace: Joint Face Detection and Alignment Using Face as Point.

Xu, Yuanyuan(Huaqiao University) and Yan, Wan(StarClouds) and Sun, Haixin(Xiamen University)

and Yang, Genke(Shanghai Jiaotong University) and Luo, Jiliang(Huaqiao University)

# [模型架构](#目录)

CenterFace 使用 mobilenetv2 作为 backbone，增加 4 层 fpn，最终通过中心点、landmark、offerset、weight_height 这 4 个 head 的权衡，得到中心点和框的位置大小以及分数。

# [数据集](#目录)

请注意，您可以基于原始论文中提到的数据集或在相关域/网络体系结构中广泛使用的数据集运行脚本。 在以下各节中，我们将在下面介绍如何使用相关的数据集运行脚本。

- 目录结构如下，目录和文件的名称由用户定义：
  ```path
      ├── dataset
          ├── centerface
              ├── annotations
              │   ├─ train_wider_face.json
              ├── images
              │   ├─ train
              │   │    └─images
              │   │       ├─class1_image_folder
              │   │       ├─ ...
              │   │       └─classn_image_folder
              │   └─ val
              │       └─images
              │           ├─class1_image_folder
              │           ├─ ...
              │           └─classn_image_folder
              ├── ground_truth
                  ├─wider_easy_val.mat
                  ├─wider_face_val.mat
                  ├─wider_hard_val.mat
                  └─wider_medium_val.mat
  ```

我们建议用户使用 WiderFace 数据集来使用我们的模型，其他数据集则需要使用与 WiderFace 相同的格式。在我们的模型中，训练集标注文件格式是 coco 格式，图像是 widerface 数据集，验证集真实标签是.mat 文件。训练集标注可以从[Baidu](https://pan.baidu.com/s/1j_2wggZ3bvCuOAfZvjWqTg)下载，密码：f9hh。图像可以从[Widerface](http://shuoyang1213.me/WIDERFACE/index.html)下载。验证集标注可以从 [ground_truth](https://github.com/chenjun2hao/CenterFace.pytorch/tree/master/evaluate/ground_truth)下载。

# [环境要求](#目录)

\- 硬件（Ascend 或 GPU）

\- 使用 Ascend 或 GPU 处理器准备硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过后，即可获得资源。

\- 框架

\- [MindSpore](https://www.mindspore.cn/install)

\- 如需查看详情，请参见如下资源：

\- [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)

\- [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# [快速入门](#目录)

通过官网安装 MindSpore 后，您可以按照以后步骤开始训练和评估：

步骤 1：准备预训练模型：通过 mindspore 训练 mobilenet_v2 或者使用以下脚本：

```python
#CenterFace need a pretrained mobilenet_v2 model:
#        mobilenet_v2_key.ckpt is a model with all value zero, we need the key/cell/module name for this model.
#        you must first use this script to convert your mobilenet_v2 pytorch model to mindspore model as a pretrain model.
#        The key/cell/module name must as follow, otherwise you need to modify "name_map" function:
#            --mindspore: as the same as mobilenet_v2_key.ckpt
#            --pytorch: same as official pytorch model(e.g., official mobilenet_v2-b0353104.pth)
python convert_weight_mobilenetv2.py --ckpt_fn=./mobilenet_v2_key.ckpt --pt_fn=./mobilenet_v2-b0353104.pth --out_ckpt_fn=./mobilenet_v2.ckpt
```

mobilenet_v2-b0353104.pth 可以从[mobilenet_v2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)下载。
若 npu 环境不兼容，可尝试在 gpu 环境运行。

步骤 2：准备 rank_table

```python
# user can use your own rank table file
# or use the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate rank table file
# e.g., python hccl_tools.py --device_num "[0,8)"
python hccl_tools.py --device_num "[0,8)"
```

步骤 3：训练

```python
cd scripts;
# prepare data_path, use symbolic link
ln -sf [USE_DATA_DIR] dataset
# check you dir to make sure your datas are in the right path
ls ./dataset/centerface # data path
ls ./dataset/centerface/annotations/train_wider_face.json # annot_path
ls ./dataset/centerface/images/train/images # img_dir
```

```python
# enter script dir, train CenterFace
# single device
bash train_standalone.sh or bash train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
# multi-device
bash train_distribute.sh or bash train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE]
# after training
mkdir ./model
cp device0/outputs/*/*.ckpt ./model # cp model to [MODEL_PATH]
```

步骤 4：测试

```python
# test CenterFace preparing
cd ../dependency/centernet/src/lib/external;
python setup.py install;
make;
cd -; #cd ../../../../../scripts;
cd ../dependency/evaluate;
python setup.py install; # used for eval
cd -; #cd ../../scripts;
mkdir ./output
mkdir ./output/centerface
# check you dir to make sure your datas are in the right path
ls ./dataset/centerface/images/val/images/ # data path
ls ./dataset/centerface/ground_truth/wider_face_val.mat # annot_path
```

```python
# test CenterFace
bash test_distribute.sh
```

步骤 5：评估

```python
# after test, eval CenterFace, get MAP
# cd ../dependency/evaluate;
# python setup.py install;
# cd -; #cd ../../scripts;
bash eval_all.sh
```

# [脚本描述](#目录)

## [脚本和样例代码](#目录)

```path
├── cv
    ├── centerface
        ├── train.py                     // training scripts
        ├── test.py                      // testing training outputs
        ├── export.py                    // convert mindspore model to air model
        ├── README.md                    // descriptions about CenterFace
        ├── scripts
        │   ├──eval.sh                   // evaluate a single testing result
        │   ├──eval_all.sh               // choose a range of testing results to evaluate
        │   ├──test.sh                   // testing a single model
        │   ├──test_distribute.sh        // testing a range of models
        │   ├──test_and_eval.sh          // test then evaluate a single model
        │   ├──train_standalone.sh       // train in ascend with single npu
        │   ├──train_distribute.sh       // train in ascend with multi npu
        ├── src
        │   ├──__init__.py
        │   ├──centerface.py             // centerface networks, training entry
        │   ├──dataset.py                // generate dataloader and data processing entry
        │   ├──config.py                 // centerface unique configs
        │   ├──losses.py                 // losses for centerface
        │   ├──lr_scheduler.py           // learning rate scheduler
        │   ├──mobile_v2.py              // modified mobilenet_v2 backbone
        │   ├──utils.py                  // auxiliary functions for train, to log and preload
        │   ├──var_init.py               // weight initilization
        │   ├──convert_weight_mobilenetv2.py   // convert pretrained backbone to mindspore
        │   ├──convert_weight.py               // CenterFace model convert to mindspore
        └── dependency                   // third party codes: MIT License
            ├──extd                      // training dependency: data augmentation
            │   ├──utils
            │   │   └──augmentations.py  // data anchor sample of PyramidBox to generate small images
            ├──evaluate                  // evaluate dependency
            │   ├──box_overlaps.pyx      // box overlaps
            │   ├──setup.py              // setupfile for box_overlaps.pyx
            │   ├──eval.py               // evaluate testing results
            └──centernet                 // modified from 'centernet'
                └──src
                    └──lib
                        ├──datasets
                        │   ├──dataset            // train dataset core
                        │   │   ├──coco_hp.py     // read and formatting data
                        │   ├──sample
                        │   │   └──multi_pose.py  // core for data processing
                        ├──detectors              // test core, including running, pre-processing and post-processing
                        │   ├──base_detector.py   // user can add your own test core; for example, use pytorch or tf for pre/post processing
                        ├──external               // test dependency
                        │   ├──__init__.py
                        │   ├──Makefile           // makefile for nms
                        │   ├──nms.pyx            // use soft_nms
                        │   ├──setup.py           // setupfile for nms.pyx
                        └──utils
                            └──image.py           // image processing functions
```

## [脚本参数](#目录)

1. 训练脚本参数

命令是：python train.py [训练参数]
train.py 主要参数如下：

```python
--lr: learning rate
--per_batch_size: batch size on each device
--is_distributed: multi-device or not
--t_max: for cosine lr_scheduler
--max_epoch: training epochs
--warmup_epochs: warmup_epochs, not needed for adam, needed for sgd
--lr scheduler: learning rate scheduler, default is multistep
--lr_epochs: decrease lr steps
--lr_gamma: decrease lr by a factor
--weight_decay: weight decay
--loss_scale: mix precision training
--pretrained_backbone: pretrained mobilenet_v2 model path
--data_dir: data dir
--annot_path: annotations path
--img_dir: img dir in data_dir
```

2. centerface 特定配置：在 config.py 中；不建议用户修改
3. 测试脚本参数：

命令是：python test.py [测试参数]

test.py 主要参数如下：

```python
test_script_path: test.py path;
--is_distributed: multi-device or not
--data_dir: img dir
--test_model: test model dir
--ground_truth_mat: ground_truth file, mat type
--save_dir: save_path for evaluate
--rank: use device id
--ckpt_name: test model name
# blow are used for calculate ckpt/model name
# model/ckpt name is "0-" + str(ckpt_num) + "_" + str(steps_per_epoch*ckpt_num) + ".ckpt";
# ckpt_num is epoch number, can be calculated by device_num
# detail can be found in "test.py"
# if ckpt is specified not need below 4 parameter
--device_num: training device number
--steps_per_epoch: steps for each epoch
--start: start loop number, used to calculate first epoch number
--end: end loop number, used to calculate last epoch number
```

4. 评估脚本参数：

命令是：python eval.py [pred] [gt]

eval.py 主要参数如下：

```python
--pred: pred path, test output test.py->[--save_dir]
--gt: ground truth path
```

## [训练过程](#目录)

### 训练

'task_set'对多 npu 训练获得更快速度是非常重要的

--task_set: 0，不 task_set；1，task_set;

--task_set_core：task_set 核数量，最大时间=cpu 的数量/每个节点的进程

步骤 1：用户需要通过 mindspore 训练一个 mobilenet_v2 模型或者使用脚本如下：

```python
python torch_to_ms_mobilenetv2.py --ckpt_fn=./mobilenet_v2_key.ckpt --pt_fn=./mobilenet_v2-b0353104.pth --out_ckpt_fn=./mobilenet_v2.ckpt
```

步骤二：准备用户 rank_table

```python
# user can use your own rank table file
# or use the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate rank table file
# e.g., python hccl_tools.py --device_num "[0,8)"
python hccl_tools.py --device_num "[0,8)"
```

步骤三：训练

- Single device

```python
# enter script dir, train CenterFace
cd scripts
# you need to change the parameter in train_standalone.sh
# or use symbolic link as quick start
# or use the command as follow:
#   USE_DEVICE_ID: your device
#   PRETRAINED_BACKBONE: your pretrained model path
#   DATASET: dataset path
#   ANNOTATIONS: annotation path
#   images: img_dir in dataset path
bash train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
# after training
cp device0/outputs/*/*.ckpt [MODEL_PATH]
```

- multi-device(推荐)

```python
# enter script dir, train CenterFace
cd scripts;
# you need to change the parameter in train_distribute.sh
# or use symbolic link as quick start
# or use the command as follow, most are the same as train_standalone.sh, the different is RANK_TABLE
#   RANK_TABLE: for multi-device only, from generate_rank_table.py or user writing
bash train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]
# after training
cp device0/outputs/*/*.ckpt [MODEL_PATH]
```

在经过 8 卡训练后，loss 值会生成如以下方式：

```python
# grep "loss is " device0/xxx.log
# epoch: 1 step: 1, loss is greater than 500 and less than 5000
2021-03-23 07:12:12,147:INFO:epoch:1, iter:0, avg_loss:loss:786.104248, loss:786.104248046875, overflow:False, loss_scale:1024.0
2021-03-23 07:12:12,350:INFO:epoch:1, iter:1, avg_loss:loss:569.002609, loss:351.9009704589844, overflow:False, loss_scale:1024.0
...
2021-03-23 07:13:01,820:INFO:epoch:2, iter:197, avg_loss:loss:1.927492, loss:1.664547324180603, overflow:False, loss_scale:1024.0
2021-03-23 07:13:02,238:INFO:epoch[2], loss:1.927492, 462.25 imgs/sec, lr:0.004000000189989805
2021-03-23 07:13:02,309:INFO:epoch:3, iter:0, avg_loss:loss:1.696034, loss:1.6960344314575195, overflow:False, loss_scale:1024.0
...
# epoch: 140 average loss is greater than 0.3 and less than 1.5:
2021-03-23 08:16:11,545:INFO:epoch:140, iter:196, avg_loss:loss:0.912602, loss:0.9277399182319641, overflow:False, loss_scale:1024.0
2021-03-23 08:16:11,675:INFO:epoch:140, iter:197, avg_loss:loss:0.911487, loss:0.6917725801467896, overflow:False, loss_scale:1024.0
2021-03-23 08:16:11,951:INFO:epoch[140], loss:0.911487, 588.05 imgs/sec, lr:3.9999998989515007e-05
2021-03-23 08:16:12,112:INFO:==========end training===============
```

模型的 checkpoint 会保存在 scripts/device0/output/xxx/xxx.ckpt。

## [测试过程](#目录)

### 测试

```python
cd scripts;
cd ../dependency/centernet/src/lib/external;
python setup.py install;
make;
cd ../../scripts;
mkdir [SAVE_PATH]
```

1. 测试单个 ckpt 文件

```python
# you need to change the parameter in test.sh
# or use symbolic link as quick start
# or use the command as follow:
#   MODEL_PATH: ckpt path saved during training
#   DATASET: img dir
#   GROUND_TRUTH_MAT: ground_truth file, mat type
#   SAVE_PATH: save_path for evaluate
#   DEVICE_ID: use device id
#   CKPT: test model name
bash test.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_ID] [CKPT]
```

2. 测试多个 ckpt 选出最好的一个

```python
# you need to change the parameter in test.sh
# or use symbolic link as quick start
# or use the command as follow, most are the same as test.sh, the different are:
#   DEVICE_NUM: training device number
#   STEPS_PER_EPOCH: steps for each epoch
#   START: start loop number, used to calculate first epoch number
#   END: end loop number, used to calculate last epoch number
bash test_distribute.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [DEVICE_NUM] [STEPS_PER_EPOCH] [START] [END]
```

在测试后，你可以发现多个保存 box 和置信度 txt 文件，打开之后你可以看到：

```python
646.3 189.1 42.1 51.8 0.747 # left top hight weight score
157.4 408.6 43.1 54.1 0.667
120.3 212.4 38.7 42.8 0.650
...
```

## [评估过程](#目录)

### 评估

```python
cd ../dependency/evaluate;
python setup.py install;
cd ../../scripts;
```

1. 评估单个测试输出

```python
# you need to change the parameter in eval.sh
# default eval the ckpt saved in ./scripts/output/centerface/140
bash eval.sh or bash eval.sh [pred] [gt]
```

2. 用户评估多个测试输出选择最好的一个

```python
# you need to change the parameter in eval_all.sh
# default eval the ckpt saved in ./scripts/output/centerface/[89-140]
bash eval_all.sh or bash eval_all.sh [pred] [gt]
```

3.  test+eval

```python
# you need to change the parameter in test_and_eval.sh
# or use symbolic link as quick start, default eval the ckpt saved in ./scripts/output/centerface/140
# or use the command as follow, most are the same as test.sh, the different are:
#   GROUND_TRUTH_PATH: ground truth path
bash test_and_eval.sh [MODEL_PATH] [DATASET] [GROUND_TRUTH_MAT] [SAVE_PATH] [CKPT] [GROUND_TRUTH_PATH]
```

你可以通过 eval.sh 得到 MAP 如下

```log
(ci3.7) [root@bms-aiserver scripts]# bash eval.sh or bash eval.sh [pred] [gt]
==================== Results = ==================== ./scripts/outputs/centerface/140
Easy   Val AP: 0.9206182787007993
Medium Val AP: 0.9173183100368525
Hard   Val AP: 0.7817584461201261
=================================================
end eval
```

你可以通过 eval_all.sh 得到 MAP 如下

```log
(ci3.7) [root@bms-aiserver scripts]# bash eval_all.sh or bash eval_all.sh [pred] [gt]
==================== Results = ==================== ./scripts/output/centerface/89
Easy   Val AP: 0.8936318136407687
Medium Val AP: 0.8979048294911587
Hard   Val AP: 0.7686040774641636
=================================================
==================== Results = ==================== ./scripts/output/centerface/90
Easy   Val AP: 0.8902698768408237
Medium Val AP: 0.892289516639168
Hard   Val AP: 0.7645320608741875
=================================================
...
==================== Results = ==================== ./scripts/output/centerface/125
Easy   Val AP: 0.9204046747187106
Medium Val AP: 0.9167227213722241
Hard   Val AP: 0.7830467644993906
=================================================
...
==================== Results = ==================== ./scripts/output/centerface/131
Easy   Val AP: 0.9219767240943944
Medium Val AP: 0.9183270693788983
Hard   Val AP: 0.7838974687232889
=================================================
...
==================== Results = ==================== ./scripts/output/centerface/140
Easy   Val AP: 0.9206182787007993
Medium Val AP: 0.9173183100368525
Hard   Val AP: 0.7817584461201261
=================================================
```

## [转换过程](#目录)

### 转换

如果你想在 A310 上推理网络，你需要把模型转换成 AIR：

```python
python export.py [ckpt_file]
```

# [模型描述](#目录)

## [性能](#目录)

### 评估性能

CenterFace 在 13K 图像上训练（标注和数据格式必须和 widerface 相同）

| 参数           | CenterFace                                                                           |
| -------------- | ------------------------------------------------------------------------------------ |
| 资源           | Ascend 910; CPU 2.60GHz, 192 核; 内核, 755G                                          |
| 上传日期       | 3/24/2021 (月/日/年)                                                                 |
| MindSpore 版本 | 1.1.1                                                                                |
| 数据集         | 13K 图像                                                                             |
| 训练参数       | epoch=140, steps=198\*epoch, batch_size=8, lr=0.004                                  |
| 损失函数       | Adam                                                                                 |
| 优化器         | Focal Loss, L1 Loss, Smooth L1 Loss                                                  |
| 输出           | heatmaps                                                                             |
| 损失           | 0.3-1.5, 最后一个 epoch 的平均 loss 范围是 0.8-1.0                                   |
| 速度           | 1p 76 img/s, 8p 460 img/s                                                            |
| 总时间         | train(8p) 1.1h, test 5min, eval 2min                                                 |
| 微调检测点     | 23.1M (.ckpt file)                                                                   |
| 脚本           | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface> |

### 推理性能

CenterFace 在 3.2K 图像上推理（标注和数据格式必须和 widerface 相同）

| 参数           | CenterFace                                     |
| -------------- | ---------------------------------------------- |
| 资源           | Ascend 910; CPU 2.60GHz, 192 核; 内核, 755G    |
| 上传日期       | 3/24/2021 (月/日/年)                           |
| MindSpore 版本 | 1.1.1                                          |
| 数据集         | 3.2K 图像                                      |
| batch_size     | 1                                              |
| 输出           | 框的位置和置信度                               |
| 正确率         | Easy 92.19% Medium 91.83% Hard 78.38% (+-0.5%) |
| 推理模型       | 23.1M (.ckpt file)                             |

# [随机情况说明](#目录)

在 dataset.py 中，我们在`create_dataset`中设置种子。在 var_init.py 中，我们为权重初始化设置了种子。

# [ModelZoo 主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
