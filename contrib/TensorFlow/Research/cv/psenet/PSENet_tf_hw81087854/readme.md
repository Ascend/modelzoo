# PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network
原始模型参考[github链接](https://github.com/liuheng92/tensorflow_PSENet),迁移训练代码到NPU

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- 其他依赖参考requirements.txt
- 数据集，下面有百度网盘下载链接，提取码1234

## 代码路径解释
```shell
.
├── checkpoint        ----存放训练ckpt的路径
├── eval.py           ----推理入口py     
├── eval.sh           ----推理shell，计算icdar2015测试集的精度、召回率、F1 Score
├── evaluation        ----精度计算相关的py，新增
├── LICENSE
├── nets              ----网络模型定义，包含backbone
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   └── resnet
├── npu_train.py      ----NPU训练
├── ocr               ----数据集存放目录
│   ├── ch4_test_images  --test图片
│   └── icdar2015        --train图片
├── pretrain_model    ----backbone
├── pse               ----后处理PSE代码
│   ├── include
│   ├── __init__.py
│   ├── Makefile
│   ├── pse.cpp
│   ├── pse.so
│   └── __pycache__
├── readme.md
├── train_npu.sh     ----NPU训练入口shell
├── train.py         ----GPU训练
└── utils            ----数据集读取和预处理
    ├── data_provider
    ├── __init__.py
    ├── __pycache__
    └── utils_tool.py
```

## 准备数据和Backbone模型
Icdar2015、Icdar2017可以去官网下载，或者直接从百度网盘里面获取，Backbone使用Resnet50_v1 [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) [BaiduYun link，提取码1234](https://pan.baidu.com/s/1gh8q0WWoqWXHHtIumUG_Mg) 

存放目录参考上面的解释。

## 一些说明
1、原始Github链接中，作者给出的预训练模型基于Icdar2015+Icdar2017数据集训练，Icdar2015测试集评估，
![输入图片说明](https://images.gitee.com/uploads/images/2021/0219/235136_f88bf050_8432352.png "屏幕截图.png")

精度数据：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.766     | 0.677  | 0.719 |

2、给出的训练超参也是基于预训练模型进行Finetune的超参：
```
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './resnet_train/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
```
3、本次实现，重新调整超参，使用Resnet50_v1预训练模型作为BackBone，使用Icdar2015和Icdar2015+Icdar2017数据集分别重新进行训练。

### NPU训练
因为第一次启动训练的时候，并不了解原作者的意图，参考原代码的训练，没有使用Backbone和预训练模型：
```
export TF_CPP_MIN_LOG_LEVEL=2
export RANK_SIZE=1
python3.7 train_npu.py \
--input_size=512 \
--batch_size_per_gpu=8 \
--checkpoint_path=./checkpoint/ \
--training_data_path=./ocr/icdar2015/
```
### TotalLoss趋势比对（NPU vs GPU）
数据集Icdar2015、超参相同时,10w个Step，NPU大概花费11小时，TotalLoss收敛趋势基本一致 :
![输入图片说明](https://images.gitee.com/uploads/images/2021/0220/000403_b5cfae72_8432352.png "屏幕截图.png")

蓝色是NPU，红色是GPU.

### 精度评估
首先确保安装依赖:
```
apt-get install zip
pip3.7 install Polygon3
```
 - 注意需根据实际python环境编辑"pse/Makefile"文件， 示例**python3.7-config**:
```
CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3.7-config --cflags)
LDFLAGS = $(shell python3.7-config --ldflags)
```
等训练10w个step结束之后，可以使用eval.sh来评估模型的精度，使用的icdar2015测试集：
```
bash eval.sh
```
Details in eval.sh：
```
export output_dir=./output
export ckpt_dir=./checkpoint/
export test_data=./ocr/ch4_test_images

mkdir ${output_dir}
rm -rf ${output_dir}/*

python3.7 eval.py \
--test_data_path=${test_data} \
--checkpoint_path=${ckpt_dir}  \
--gpu_list=""   \
--output_dir=${output_dir}

cd ${output_dir}
zip results.zip res_img_*.txt
cd ../

python3.7 evaluation/script.py -g=./evaluation/gt.zip -s=${output_dir}/results.zip
```
### 精度、召回率、F1 Score比对（NPU vs GPU）:
相同的训练集 icdar2015 (1000 images) 和测试集(500 images)([BaiduYun link，提取码1234](https://pan.baidu.com/s/12qlSPPZl2a8rAIqeMAMyUA) 
) 和相同的超参，不加载Backbone的情况下，虽然Loss值收敛趋势一致，但是精度都很低：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.460     | 0.355  | 0.401 |
| NPU | 0.459     | 0.371  | 0.410 |

NPU和GPU的Events文件（ckpt精度太低就不存放了）([BaiduYun link，提取码1234](https://pan.baidu.com/s/1rxc7WabhieyXYvaNM9pQUw）)

### 精度提升
发现精度与预期相差太大之后，阅读原作者的代码，发现应该要带上BackBone的预训练模型进行训练：

icdar2015数据集，超参如下：
```
export TF_CPP_MIN_LOG_LEVEL=2
export RANK_SIZE=1
python3.7 train_npu.py \
--input_size=512 \
--batch_size_per_gpu=8 \
--checkpoint_path=./checkpoint/ \
--training_data_path=./ocr/icdar2015/ \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
得到精度：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| NPU | 0.606     | 0.621  | 0.613 |

icdar2015+icdar2017数据集[BaiduYun link，提取码1234](https://pan.baidu.com/s/1bmKVFuiDTpWg_TiYLDhnMg)\
NPU训练约15hours，超参如下：
```
export TF_CPP_MIN_LOG_LEVEL=2
export RANK_SIZE=1
python3.7 train_npu.py \
--input_size=512 \
--batch_size_per_gpu=8 \
--checkpoint_path=./checkpoint/ \
--training_data_path=./ocr/icdar2015/ \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
得到精度：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| NPU | 0.696     | 0.609  | 0.649 |

最后调整其他超参，NPU上训练得到精度超过原作者提供的预训练模型：
```
export TF_CPP_MIN_LOG_LEVEL=2
export RANK_SIZE=1
python3.7 npu_train.py \
--input_size=512 \
--learning_rate=0.0001 \
--batch_size_per_gpu=14 \
--num_readers=16  \
--checkpoint_path=./checkpoint/ \
--training_data_path=./ocr/icdar2015/ \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
TotalLoss趋势比对（NPU vs GPU）：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0220/002637_fe59b040_8432352.png "屏幕截图.png")

红色是GPU，绿色是NPU

精度比对：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.764     | 0.714  | 0.738 |
| NPU | 0.730     | 0.742  | 0.736 |

GPU和NPU的ckpt&events：[BaiduYun link，提取码1234]（https://pan.baidu.com/s/1oDi54CifBtWVIp6XsFluFQ)