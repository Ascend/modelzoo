# JASPER for ACL

## 目录

- [目的](#目的)
- [环境安装依赖](#环境安装依赖)
- [数据集及模型准备](#数据集及模型准备)
- [模型转换](#模型转换)
- [demo编译](#demo编译)
- [离线推理](#离线推理)
- [在线推理](#)

   

## 目的

- 基于Jsaper网络进行英文音频识别。

## 环境安装依赖

- 参考《驱动和开发环境安装指南》安装NPU训练环境
- python依赖库：pandas、reasmpy、h5py、numpy
- TensorFlow>=1.10

## 数据集及模型准备

Jasper NPU训练代码参考：

[训练代码参考](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/audio/Jasper_for_TensorFlow)

获取在线推理的数据集目录下的文件，拷贝到./datasets/目录下

pb模型：

获取jsaper checkpoint转pb的pb模型，将pb模型放在model目录下，checkpoint转pb参考脚本：

```
cd generate_pb
python ckpt2pb.py
```



## 模型转换

离线模型转换

```
cd model
./Jasper_convert_om_fp32.sh --batchSize=1 --soc_version=Ascend310
```

该脚本用来做Jasper模型的atc转换，支持2个参数：

```
--batchSize			离线推理模型的batchSize，默认值为1
--soc_version		离线推理的soc_version，对于1910，这个参数设置为Ascend310,对于1980,这个参数设置为Ascend910A,默认值为Ascend310
模型名称固定为：jasper_infer_float32.pb,可以根据实际生成的模型名称修改
```



## demo编译

如果demo编译后在host侧运行，则执行以下脚本:

```
cd benchmark_infer
chmod +x build_host.sh
./build_host.sh
```

如果demo编译后在device侧运行，则执行以下脚本

```
cd benchmark_infer
chmod +x build_device.sh
./build_device.sh
```

可执行文件生成在../output目录

## 离线推理

端到端拉起离线推理

```
./benchmark_jasper_tf.sh --preprocess=0 --json=jasper_syn_inference_b1_FP32.json
```

脚本中包含预处理，后处理和推理

### 预处理脚本

```
cd pre_treatment
./prerun_jasper_infer_input_fp32.sh

--datasets_folder  预处理数据集路径，默认值为./datasets
--output_folder	   预处理的输出路径，默认值为./datasets/jasper
```

### 推理脚本

```
./benchmark jasper_syn_inference_b1_FP32.json
```

json脚本中，涉及om_path_list，dir_path_list，result_path_list，json文件中给出的只是样例，建议根据环境上实际的路径替换修改

数据结果路径：./model1_dev_0_chn_0_result/jasper/

精度数据：./output/perform_static_dev_0_chn_0.txt

### 后处理脚本

```
cd post_treatment
./jasper_accuracy_calc_fp32.sh

--inferRet_folder	推理输出的结果文件路径，默认值为：./model1_dev_0_chn_0_result/jasper/
--real_file			真值文件路径，默认值为:./datasets/librivox-dev-clean.csv
```

精度数据生成在：./output/predict_accuracy.txt文件中



## 在线推理

执行在线推理：

```
cd output
python3.7.5 jasper_online_inference.py

支持以下参数:
--model_path	pb模型的路径，默认值为../model/jasper_infer_float32.pb
--data_dir		原始dev.json文件的目录，默认值为：../datasets
--output_dir	预处理后的jasper	bin文件的输出目录../datasets
--pre_process  	是否做预处理，True or False，默认值False
--post_process  是否做后处理，True or False，默认值True
--batchSize		在线推理的batch数，默认值为1
```







