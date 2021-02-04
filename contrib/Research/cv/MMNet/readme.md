# MMNet
## 模型简介

## 结果


## Requirements

---
## 数据准备
### 预训练模型

```  
### 数据集下载


### Fintuned 模型
 
```


在GPU上复现的模型在
```

```

## 训练
### 参数说明

### 运行命令


## 测试 
### 参数说明


### 运行命令


## 离线推理
### 1、原始模型转PB模型
```
bash scripts/run_conver_ckpt2pb.sh path/to/dataset path/to/ckpt

# example:
# bash scripts/run_conver_ckpt2pb.sh /home/nankai/dataset/mmnet results/train/MMNetModel-1593500
```
sh文件后需要指定数据集位置和ckpt模型文件路径，转换的pb模型会保存在offline_infer文件夹中，model.pb为图文件，mmnet.pb为网络静态模型文件，最终使用mmnet.pb 其输出节点被构建为'output'.
我们转换好的PB模型在 xxx 提取码: xxxx

### 2、PB模型转OM模型
使用HECS上的Mind-Studio转换PB模型到OM模型, 选择算子融合，以加速静态模型的运行速度，其输入格式为NHWC, 输出节点为output，其中后处理为图像的步骤已经包含在模型文件中。
我们转换好的OM模型在 xxx 提取码: xxxx

### 3、数据预处理
读取数据集中所有图片，对其进行预处理，默认保存在offline_infer的Bin/test文件夹中d的images和masks
```
bash scripts/run_datapre.sh path/to/dataset
#example:
# bash scripts/run_datapre.sh /home/nankai/dataset/mmnet 
```
或者从 ```OBS://...``` 下载并解压到```./offline_infer/```目录下

### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/ccl/msame)

### 5、推理性能精度测试
#### 推理性能测试
使用如下命令进行性能测试：
```
./msame --model ./mmnet.om --output ./output/ --loop 100
```
测试结果如下：
```
To do
```
Batchsize=1, input shape = [1, 256,256, 3], 平均推理时间xxx ms

#### 推理精度测试
使用OM模型推理结果，运行：
```
.//msame --model ./offline_infer/mmnet.om --input ./offline_infer/Bin/test/images --output ./offline_infer/Bin/test/outputs
```
所有的输出会保存在```./offline_infer/Bin/test/output```目录下，
或者从 ```OBS://``` 下载并解压到该目录下

运行以下命令进行离线推理精度测试, 默认加载上面保存的默认目录中的output和masks.
```
python offline_infer/evaluate.py 
```
离线推理精度MAD=...，与在线模型精度一致
