# Two-Stream Inflated 3D ConvNet (I3D)

## 简介

* [模型的来源及原理](#模型的来源及原理)
* [模型复现的步骤](#模型复现的步骤)
  * [创建数据列表文件](#创建数据列表文件)
  * [下载预训练模型](#下载预训练模型)
  * [训练](#训练)
  * [测试](#测试)
  * [外部库](#外部库)
  * [结果](#结果)


## 模型的来源及原理

- Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset是2017年发表在CVPR上的论文，原论文：[I3D paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf)，源码：[code](https://github.com/USTC-Video-Understanding/I3D_Finetune)。
- I3D论文为了实现行为识别，提出了Kinetics数据集，并设计了I3D模型，使用ImageNet和Kinetics上预训练的参数，在UCF101等标准数据集上达到了当时最佳的效果。

## 模型复现的步骤

### 1.创建数据列表文件
源码通过遍历 `/data/ucf101/` 中的rgb.txt和flow.txt文件，查找磁盘上的 `rgb` 数据和 `flow` 数据，这里将 `rgb` 数据和 `flow` 数据放在了OBS桶中。

 **下载链接** 如下：[rgb](https://i3d-ucf101.obs.cn-north-4.myhuaweicloud.com/jepgs_256.zip) [flow](https://i3d-ucf101.obs.cn-north-4.myhuaweicloud.com/ucf101_tv1_flow.zip)。下载完成后，需要将两个文件放置在 `/data/` 目录下


我们必须调整列表文件，以确保列表文件能够映射数据的正确路径。具体来说，对于rgb数据，必须更新 `/data/ucf101/rgb.txt`。该文件中的每一行应采用以下格式:

```
dir_name_of_imgs_of_a_video /path/to/img_dir num_imgs label 
```

例如，如果你的UCF101的RGB数据保存在 '/data/ucf101/jpegs_256' 中，这个文件夹中有13320个子文件，每个子文件都包含一个视频的图片。如果子文件 `v_BalanceBeam_g14_c02`中有`96`张图片，并且该视频的`label=4`，则此子文件在rgb.txt中的保存格式为:

```
v_BalanceBeam_g14_c02 /data/jpegs_256/v_BalanceBeam_g14_c02 96 4
```

同理，flow数据文件的格式为：

```
v_Archery_g01_c06 /data/tvl1_flow/{:s}/v_Archery_g01_c06 107 2
```
更新`rgb.txt和flow.txt`文件 **运行**  

```
create_data_to_txt.py --rgb_path=../data/jpegs_256 --rgb_save_path=../data/rgb.txt --flow_path=../data/tvl1_flow --flow_save_path=../data/flow.txt --label_map_path=../data/ucf101/label_map.txt
```
其中 'rgb_path'为rgb类型数据的存储位置，'rgb_save_path'为rgb.txt文件的存储位置，'flow_path'为flow类型数据的存储位置，'flow_save_path'为flow.txt文件的存储位置，'label_map_path'为ucf101数据的label_map.txt文件的存储目录


### 2.下载预训练模型
为了在ucf101上优化I3D网络，需要在下载DeepMind提供的Kinetics预训练I3D模型。为了方便，我们将预训练模型放在了OBS桶中，

下载：[checkpoints](https://i3d-ucf101.obs.cn-north-4.myhuaweicloud.com/checkpoints.zip)  具体来说，下载`checkpoints` 并将其放在 `/data` 目录下:

### 3. 在ucf101训练

```
# Finetune on split1 of rgb data of UCF101
python3.7 finetune_new_bn.py --dataset=ucf101 --mode=rgb --split=1
# Finetune on split2 of flow data of UCF101
python3.7 finetune_new_bn.py --dataset=ucf101 --mode=flow --split=2 
```
### 4.test

待完善

### 5. 外部库

运行代码需要安装：

sonnet： pip install dm-sonnet==1.23
pillow： pip install pillow

### 6. 结果
原论文结果：
| Training Split | rgb | flow | fusion |
|----------------|-----|------|--------|
|     Split1     |94.7%|96.3% | 97.6%  |

目前结果（待完善）：

| Training Split | rgb | flow | fusion |
|----------------|-----|------|--------|
|     Split1     |96.2%|    % |        |

rgb为88.3%时dropout=0.5  step: 9540, loss: 0.0786, accuracy: 0.883