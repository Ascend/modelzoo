## deeplab v2

## 概述
迁移deeplabv2到ascend910平台上使用NPU运行，并将结果与原论文进行对比
| Accuracy | Paper | Ours  |
|----------|-------|-------|
| mIoU     | 0.715 | 0.744 |

## Requirements

1.Tensorflow 1.15
2.Ascend910

## 代码及路径解释

- - deeplabv2
- - - README.md
- - - LICENSE
- - - dataset 用于存放训练时需要用到的训练集，验证集，测试集标签
- - - - train.txt
- - - - val.txt
- - - - test.txt
- - - testcase 用于存放自测样例标签
- - - - train.txt
- - - - val.txt
- - - - test.txt
- - - model 用于存放预训练模型 obs//deeplab-zjw/dataset/pretraind_model/deeplab_resnet.ckpt
- - - utils 用于存放数据预处理文件
- - - - __init__.py
- - - - image_reader.py
- - - - label_utils.py
- - - - write_to_log.py
- - - main.py 执行主函数代码
- - - model.py 定义模型train，eval过程的逻辑操作
- - - network.py 搭建网络结构


