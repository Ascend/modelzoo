# db模型使用说明

## 1. Requirements
* NPU配套的run包安装(C20B030)
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)

- 安装python依赖包

```
pip3.7 install -r requirements.txt
```



## 2. Models

下载预训练模型MLT-Pretrain-Resnet50, [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG). 放在文件夹path-to-model-directory下。

```
__ path-to-model-directory
  |__ MLT-Pretrain-ResNet50
```



## 3. Dataset Prepare

下载icdar2015数据集，放在文件夹datasets下。

```
__ datasets
  |__icdar2015
```

## 4. Environment setup

加载环境变量文件

```
source env.sh
```



## 5. 1P

按需要编辑device_list，运行run.sh
以下是db的1p训练脚本

```
export PYTHONPATH=./:$PYTHONPATH

python3.7 train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-tomodel-directory/MLT-Pretrain-Resnet50 \
        --seed=515 \
        --device_list "0"
```

**注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查第二章节的设置，以免影响精度。

## 6. 8P

运行run8p.sh

```
export PYTHONPATH=./:$PYTHONPATH

python3.7 -W ignore train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-tomodel-directory/MLT-Pretrain-Resnet50 \
        --seed=515 \
        --distributed \
        --device_list "0,1,2,3,4,5,6,7" \
        --num-gpus 8 \
        --local_rank 0 \
        --addr $(hostname -I |awk '{print $1}') \
        --dist_backend 'hccl' \
		--world_size 1 \ 
		--Port 29501 \
		--batch_size 128 \
		--lr 0.056
```

