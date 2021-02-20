# DB_ResNet50

# Introduction
这是DB_ResNet50模型的tensorflow版本实现，相关论文为"Real-time Scene Text Detection with Differentiable Binarization"，该版本的代码适配于Ascend910, 当前复现精度 F值0.749

# 代码及路径解释
```
.
└── dbresnet_tf_ihongming
    ├── config 存放配置文件和配置解析模块
    │   ├── base_totaltext.yaml
    │   ├── config.py
    │   └── db_config.py
    ├── datasets 用于存放训练集，验证集，测试集的标签
    │   └── total_text
    ├── data  数据预处理和数据生成器
    │   ├── augmenter.py
    │   ├── generator_enqueuer.py
    │   ├── generator.py
    │   ├── image_dataset.py
    │   └── processes 
    │       ├── augment_data.py
    │       ├── data_process.py
    │       ├── filter_keys.py
    │       ├── ...
    ├── networks 网络层
    │   ├── learning_rate.py
    │   ├── losses.py
    │   ├── model.py
    │   ├── ...
    ├── postprocess 后处理模块
    │   ├── ckpt2pb.py
    │   ├── post_process.py
    │   └── utils.py
    ├── tools 用于数据dump时生成去除随机性的tfrecord
    │   ├── gen_no_random.py
    │   └── test_no_random.py
    ├── README.md
    ├── requirements.txt
    ├── train.py 训练代码
    └── train_testcase.sh
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
你可以根据下方的[OBS链接](https://dbnet.obs.cn-north-4.myhuaweicloud.com:443/totaltext.zip?AccessKeyId=NMZTI8AEEE9WMDLZ4ABS&Expires=1606984613&Signature=yXloRdpX6hLfsdDsduBG0PBuAFs%3D)下载数据集, 并将下载好数据集后解压到 ./datasets/目录下，以下是数据解压后的一个示例。
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```

# Train
在./config 目录下有名为 **base_totaltext.yaml**,请先确认data_dir 和 data_list的路径是否正确，执行
```
python3.7 train.py -p NPU 2>&1 | tee train_log.log
```
主要参数注释：
```
max_steps 总的训练步数
save_steps 每多少步保存模型
learning_rate 初始learning_rate
platform 执行平台 NPU/GPU
```
其他的参数请在./config/db_config.py中进行修改。生成的日志和ckpt在 ./logs中，tensorboard在 ./logs/tf_logs/train中，您可以在训练过程中进行监督

# Evaluate
执行下述代码，测试最新模型的 P\R\F value
```
python3.7 eval/evaluation.py
```
