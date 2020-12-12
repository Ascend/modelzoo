
# Introduction
这是DB_ResNet50模型的tensorflow版本实现，相关论文为"Real-time Scene Text Detection with Differentiable Binarization"，该版本的代码适配于Ascend910, 当前复现精度 F值0.749

# Code path
```shell
├── README.md
├── config 配置文件处理
│   ├── __init__.py
│   ├── config.py yaml 文件解析模块
│   └── db_config.py 超参设置文件
├── data  数据预处理、数据读取模块
│   ├── __init__.py
│   ├── augmenter.py
│   ├── data_loader.py
│   ├── ...
│   └── unpack_msgpack_data.py
├── datasets 数据集
├── experiments 数据预处理模块的yaml文件
│   ├── base.yaml
│   └── base_totaltext.yaml
├── networks 网络层
│   ├── __init__.py
│   ├── losses.py
│   ├── model.py
│   ├── resnet_utils.py
│   └── resnet_v1.py
├── postprocess 后处理模块
│   ├── __init__.py
│   ├── post_process.py
│   └── utils.py
├── requirements.txt
├── requirements_all.txt
├── inference.py  样本推理
├── evalution.py  模型评估
├── DetectionIoUEvaluator.py IOU计算
├── start_train.sh  训练启动脚本
└── train.py  训练代码
```

# Installation
在Apulis docker环境下，执行
```
pip install -r requirements.txt
sudo apt-get install libgeos-dev -y
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
如果运行中找不到包，请参考requirement_all.txt内的对应版本进行下载

# Dataset
你可以根据下方的[OBS链接](https://dbnet.obs.cn-north-4.myhuaweicloud.com:443/totaltext.zip?AccessKeyId=NMZTI8AEEE9WMDLZ4ABS&Expires=1606984613&Signature=yXloRdpX6hLfsdDsduBG0PBuAFs%3D)下载数据集, 并将下载好数据集后解压到 DB/datasets/目录下，以下是数据解压后的一个示例。
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```

# Train
在DB/experiments 目录下有名为 **base*.yaml**,请先确认data_dir 和 data_list的路径是否正确，执行
```
python3.7 train.py
```
生成的日志和ckpt在 DB/logs中，tensorboard在DB/logs/tf_logs/train中，您可以在训练过程中进行监督

# Evaluate
执行下述代码，测试最新模型的 P\R\F value
```
python3.7 evaluation.py
```
