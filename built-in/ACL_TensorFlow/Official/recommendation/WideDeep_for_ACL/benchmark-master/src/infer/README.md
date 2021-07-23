# 推理benchmark

[TOC]

## 支持的产品

Atlas 300 (Model 3010)

## 支持的版本

1.70.0.0.B070

## 操作系统

Ubuntu 16.04

## 编译方法

进入build目录，运行build_local.sh脚本：

```
cd build
./build_local.sh
```

## 运行方法

编译成功之后会在工程根目录自动生成bin文件夹，进入bin并带参数运行benchmark，包含如下参数：

| 参数             | 含义                                                         |
| ---------------- | ------------------------------------------------------------ |
| -model_type      | 模型列表，当前仅支持vision/nmt/widedeep                      |
| -batch_size      | 模型的batch size                                             |
| -device_id       | 运行的npu设备编号                                            |
| -om_path         | om文件路径                                                   |
| -input_text_path | 输入文本文件路径,可为vision模型的输入图像配置文件，或者nmt/widedeep模型的输入文件 |
| -input_vocab     | 输入语言词典文件路径（针对NMT等语言翻译模型）                |
| -ref_vocab       | 输出语言词典文件路径（针对NMT等语言翻译模型）                |

例如：

运行batch size为512的widedeep模型的benchmark：

```
./benchmark -model_type=widedeep -batch_size=512 -device_id=0 -om_path=./wide_deep_bs512.om -input_text_path=./adult.test
```

