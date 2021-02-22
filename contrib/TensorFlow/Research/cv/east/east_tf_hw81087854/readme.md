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
`-- training_samples
```

## 准备数据和Backbone模型
Icdar2015、Icdar2013可以去官网下载，或者直接从百度网盘里面获取，Backbone使用Resnet50_v1 [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) 

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
bash train_npu.sh
```

### TotalLoss趋势比对（NPU vs GPU）
数据集和超参相同时:
```
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
相同的训练集 icdar2015 (1000 images) 和测试集(500 images)([BaiduYun link，提取码1234](https://pan.baidu.com/s/12qlSPPZl2a8rAIqeMAMyUA) 
) 和相同的超参，NPU的精度优于GPU（看Hmean，即为F1 Score）:
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.826     | 0.771  | 0.797 |
| NPU | 0.834     | 0.767  | 0.799 |

NPU Checkpoints: ([BaiduYun link，提取码1234](https://pan.baidu.com/s/19qRk67W3R4x_5wDbPwmWIA) )\
GPU Checkpoints: ([BaiduYun link，提取码1234](https://pan.baidu.com/s/1k77-11IJUBpXC90FpIoaqA) )

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
相同的数据集：训练集 （icdar2013+icdar2015: 229+1000 images) 和测试集(icdar2015: 500 images) ([BaiduYun link，提取码1234](https://pan.baidu.com/s/1DsEqwvOagZRadPWAyZKhUw) 
) 和相同的超参:

|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.842     | 0.766  | 0.803 |
| NPU | 0.853     | 0.773  | 0.811 |

NPU Checkpoints: ([BaiduYun link，提取码1234](https://pan.baidu.com/s/1jVMvmWgKrj2hOkvV2_0VOw) )\
GPU Checkpoints: ([BaiduYun link，提取码1234](https://pan.baidu.com/s/1dfZj6dgoQhqCrcNB9jBRvQ) )

### 图片测试
使用eval.py可以测试你自己的图片
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```