 **介绍**  
Ascend预置数据集和Python第三方库

 **一、预置数据集** 

Shell脚本中，访问时用提供变量即可

| 领域  | 数据集名称        | 访问变量            |
|-----|--------------|---------------|
| CV  | ImageNet2012_train | $ImageNet2012_train |
| CV  | ImageNet2012_val | $ImageNet2012_val |
| CV  | CIFAR10      | $CIFAR10      |
| NLP | Wikipedia_CN | $Wikipedia_CN |
| NLP | wmt-ende     | $WMT_END      |
|     |              |               |

持续建设中，若有新增需求，请提交ISSUE，标题注明[新增数据集]，内容写上数据集名称和下载地址

 **二、Python第三方库** 

安装第三方库依赖使用"pip3"、"pip3.7"，已安装的库：
```
Package               Version
--------------------- --------
absl-py               0.11.0
astor                 0.8.1
cached-property       1.5.2
cycler                0.10.0
gast                  0.2.2
google-pasta          0.2.0
grpcio                1.33.2
h5py                  3.1.0
importlib-metadata    2.0.0
Keras                 2.3.1
Keras-Applications    1.0.8
Keras-Preprocessing   1.1.2
kiwisolver            1.3.1
Markdown              3.3.3
matplotlib            3.3.3
numpy                 1.19.4
opencv-contrib-python 4.4.0.46
opencv-python         4.4.0.46
opt-einsum            3.3.0
Pillow                8.0.1
pip                   20.2.4
protobuf              3.14.0
pyparsing             2.4.7
python-dateutil       2.8.1
PyYAML                5.3.1
scipy                 1.5.4
setuptools            41.2.0
six                   1.15.0
tensorboard           1.15.0
tensorflow            1.15.0
tensorflow-estimator  1.15.1
termcolor             1.1.0
Werkzeug              1.0.1
wheel                 0.35.1
wrapt                 1.12.1
zipp                  3.4.0
```

