# EAST: An Efficient and Accurate Scene Text Detector
原始模型参考[github链接](https://github.com/argman/EAST),迁移训练代码到NPU

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- 其他依赖参考requirements.txt
- 数据集，下面有百度网盘下载链接，提取码1234

## 代码路径解释
```shell
.
|-- LICENSE
|-- __init__.py
|-- checkpoint            ----存放训练ckpt的路径
|-- data_util.py          ----训练数据处理，多进程
|-- demo_images           ----样例图片
|-- deploy.sh
|-- eval.py               ----推理入口py
|-- eval.sh               ----推理shell，计算icdar2015测试集的精度、召回率、F1 Score
|-- evaluation            ----精度计算相关的py，新增
|-- icdar.py              ----icdar数据集处理，返回图片+bbox
|-- lanms                 ----nms组件
|-- locality_aware_nms.py
|-- model.py              ----模型定义
|-- multigpu_train.py     ----GPU训练
|-- nets
|-- npu_train.py          ----NPU训练
|-- ocr                   ----数据集目录
|   |-- ch4_test_images   ----测试集
|   `-- icdar2015         ----训练集
|-- output                ----输出目录
|-- pretrain_model        ----backbone
|-- readme.md
|-- requirements.txt
|-- run_demo_server.py
|-- static
|-- templates
|-- train_npu.sh          ----NPU训练入口shell
|-- train_testcase.sh
|-- test  
    |--train_performance_1p.sh     ----CI执行入口
`-- training_samples
```

## 准备数据和Backbone模型
Icdar2015、Icdar2013可以去官网下载，Backbone使用Resnet50_v1 [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) 

存放目录参考上面的解释


## NPU训练
在NPU上面，启动训练，使用下面的命令:
```
export RANK_SIZE=1
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
或者直接执行shell:
```
bash train_npu.sh  或者  bash ./test/train_performance_1p.sh(仅200个step的调试脚本)
```

### TotalLoss趋势比对（NPU vs GPU 型号为t4）
数据集和超参相同时:
```
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
10w个Step，NPU大概花费10小时，TotalLoss收敛趋势基本一致 :\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0114/232451_0023bcbd_8432352.png "屏幕截图.png")

蓝色是NPU，红色是GPU.

### 精度评估
首先确保安装依赖:
```
apt-get install zip
pip3.7 install Polygon3
```
 - 注意需根据实际python环境编辑"lanms/Makefile"文件， 示例**python3.7-config**:
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

mkdir  ${output_dir}
rm -rf ${output_dir}/*

python3.7 eval.py \
--test_data_path=${test_data} \
--checkpoint_path=${ckpt_dir} \
--output_dir=${output_dir}

cd ${output_dir}
zip results.zip res_img_*.txt
cd ../

python3.7 evaluation/script.py -g=./evaluation/gt.zip -s=${output_dir}/results.zip
```

### 精度、召回率、F1 Score比对（NPU vs GPU）:
训练集 icdar2015 (1000 images) 

测试集icdar2015(500 images)
和相同的超参，NPU的精度优于GPU（看Hmean，即为F1 Score）:

|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU T4 | 0.826     | 0.771  | 0.797 |
| NPU | 0.834     | 0.767  | 0.799 |

### 使用Icdar2013+Icdar2015训练
需要注意到，原始的github实现中，使用的icdar2013+icdar2015数据集进行训练，所以尝试增加icdar2013训练集：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0131/214657_f59d66b0_8432352.png "屏幕截图.png")
原作者提供预训练模型，精度评估，F1 Score为0.8076：
```
{"precision": 0.8464379947229551, "recall": 0.772267693789119, "hmean": 0.8076535750251763, "AP": 0}
```
Icdar2013数据集标签格式转换：

因为Icdar2013的标签bbox格式为[xmin,ymin,xmax,ymax]，需要将标注格式转换为[x1,y1,x2,y2,x3,y3,x4,y4]格式才可以在使用，可以使用下面的脚本转换：
```
#需要根据实际的路径编辑修改下py文件里面的标签路径
python3 convert_gt_icdar2013_to_2015.py
```
或者可以直接下载下面百度云盘里面的数据集，已经做好格式转换。


GPU、NPU使用相同的超参:
```
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
从头训练，10w个Step，NPU大概花费12小时，TotalLoss收敛趋势基本一致 : 

![输入图片说明](https://images.gitee.com/uploads/images/2021/0118/233452_f06f1fb1_8432352.png "屏幕截图.png")

蓝色是NPU，红色是GPU.

#### 精度、召回率和F1 Score对比:
相同的数据集：

训练集 （icdar2013+icdar2015: 229+1000 images) 

测试集(icdar2015: 500 images) 

|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU T4| 0.842     | 0.766  | 0.803 |
| NPU | 0.853     | 0.773  | 0.811 |

### 精度、性能调优
相同超参下：
```
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=20 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=30 \
--max_steps=200000 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
#### 训练性能
NPU训练性能：\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0603/230657_ee461c68_8432352.png "屏幕截图.png")

GPU V100训练性能：\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0603/230724_c10c6187_8432352.png "屏幕截图.png")

训练性能对比：
| 平台     | BatchSize | 训练性能(imgs/s) |
|----------|---|--------------|
| NPU      | 20 |      45        |
| GPU V100 | 20 |      36        |

#### NPU AutoTune之后性能
训练时开启AutoTune：
```
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=20 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=30 \
--max_steps=200000 \
--auto_tune=True   \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
NPU训练性能：\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0623/223435_50d7e1a1_8432352.png "屏幕截图.png")
| 平台     | BatchSize | 训练性能(imgs/s) |
|----------|---|--------------|
| NPU      | 20 |      47        |

#### 在线推理性能
NPU：\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0531/232155_9aabb889_8432352.png "屏幕截图.png")

GPU V100:\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0531/232235_a162405e_8432352.png "屏幕截图.png")

推理性能对比：
| 平台       | BatchSize | TimeCost(ms) | Throughput(fps) |
|----------|-----------|--------------|-----------------|
| NPU      | 1         | 14           | 71.42           |
| GPU V100 | 1         | 20           | 50              |


#### 精度对比
GPU v100上面有个已知新Cuda版本的精度问题，训练出来的模型F1值才0.5，这里使用之前T4的精度数据对比, \
NPU新训练出来的模型+优化了NMS参数，训练20万步，F1精度由0.811提升至0.825：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU T4| 0.842     | 0.766  | 0.803 |
| NPU | 0.859     | 0.794  | 0.825 |

在模型训练的过程中，可以执行**python3 get_best_ckpt.py**,脚本会以100s为间隔轮询，取最新的ckpt在NPU/CPU/GPU上面验证测试集精度，并保存最优者。

### 预处理优化
#### 数据集预处理
原有的训练代码逻辑，是起多个线程，一边训练的同时一边进行数据预处理，然后喂给训练的session，这里根据空间换时间的思路进行优化：

先调用**icdar.py**原有接口，预处理好图片，得到按照Batchsize划分的1000组预处理好的数据，
包括resized_image、score_map、geo_map、mask：
```
#此处参数要与训练的参数保持一致
python3.7 preprocess_datasets.py \
--input_size=512 \
--batch_size_per_gpu=20 \
--num_readers=30  \
--processed_data=./processed_dataset/ \
--training_data_path=./ocr/icdar2015/

```
#### 训练时直接加载处理后的数据
```
python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=20 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=30 \
--max_steps=100000 \
--use_processed_data=True  \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
或者执行**bash train_npu_preload.sh**, 可以看到训练性能大幅提升：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0707/224939_2ce722c6_8432352.png "屏幕截图.png")

与GPU上面使用相同的预处理优化，性能对比：
| 平台     | BatchSize | 训练性能(imgs/s) |
|----------|---|--------------|
| NPU      | 20 |      87        |
| GPU V100 | 20 |      56        |

在模型训练的过程中，可以执行**python3 get_best_ckpt.py**,脚本会以100s为间隔轮询，取最新的ckpt在NPU/CPU/GPU上面验证测试集精度，并保存最优者：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| NPU| 0.840     | 0.769  | 0.803 |

### 图片测试
使用eval.py可以测试你自己的图片
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```