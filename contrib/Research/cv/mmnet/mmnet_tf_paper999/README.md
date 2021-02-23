# MMNet
## 模型简介
MMNet致力于解决移动设备上人像抠图的问题，旨在以最小的模型性能降级在移动设备上获得实时推断。MMNet模型基多分支dilated conv以及线性bottleneck模块，性能优于最新模型，并且速度提高了几个数量级。在小米Mi 5设备上，模型可以加速四倍以达到30 FPS。在相同条件下，MMNet相比于Mobile DeepLabv3模型，参数数量少一个数量级，速度更快，同时保持可比的性能。
![](./figure/gradient_error_vs_latency.png)

## 结果
<table>
    <tr>
        <td></td>
        <td>Gradient(1e-3) &#8595</td>
        <td>MAD(1e-2) &#8595</td>
    </tr>
    <tr>
        <td>Report in paper</td>
        <td>2.88</td>
        <td>2.47</td>
    </tr>
    <tr>
        <td>Reproduce on GPU</td>
        <td>4.23</td>
        <td>2.31</td>
    </tr>
    <tr>
        <td>Reproduce on Ascend 910</td>
        <td>13.26</td>
        <td>21.64</td>
    </tr>
</table>

## Requirements
- Tensorflow 1.15.0
- Ascend 910
- humanfriendly
- overload
- deprecation
---
## 数据准备
### 数据集下载
请从该[项目主页](https://1drv.ms/u/s!ApwdOxIIFBH19Ts5EuFd9gVJrKTo)下载数据集并将其解压到当前目录下

### GPU复现的模型
```
百度网盘：
链接: https://pan.baidu.com/s/11RGrZ7ByovcXSCunDO9JlA 
提取码: 68kp

或OBS链接：
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/gpu/checkpoint
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/gpu/MMNetModel-1593500.data-00000-of-00001
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/gpu/MMNetModel-1593500.meta
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/gpu/MMNetModel-1593500.index
```

### NPU复现的模型
``` 
百度网盘：
链接: https://pan.baidu.com/s/1Rs0m-QbzLDlowJ_M5oGJAA 
提取码: 5sbi

或OBS链接：
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/npu/checkpoint
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/npu/MMNetModel-2124500.data-00000-of-00001
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/npu/MMNetModel-2124500.meta
https://mmnet.obs.cn-north-4.myhuaweicloud.com/pretrained/npu/MMNetModel-2124500.index
``` 

## 训练
### 参数说明
```
--checkpoint_path 加载预训练的模型路径
--train_dir 保存checkpoint的文件夹路径
--batch_size 训练的batch_size大小
--dataset_path 数据集文件夹路径
--learning_rate 学习率大小
--step_save_checkpoint 保存checkpoint的迭代间隔
--augmentation_methodmax_to_keep 数据增广方法
--max_epoch_from_restore 加载模型后最大训练epoch数
```
### 运行命令
```
sh ./scripts/train.sh /path/to/dataset /path/to/training/directory
```
例：
```
sh ./scripts/train.sh ./dataset/ ./results/train/
```

## 测试 
### 参数说明
```
--dataset_path 数据集文件夹路径
--checkpoint_path 需要测试的checkpoint的路径
```

### 运行命令
```
sh ./scripts/validate.sh /path/to/dataset /path/to/training/directory
```
例：
```
sh ./scripts/validate.sh ./dataset/ ./results/train/
```

## 离线推理
### 1、原始模型转PB模型
```
bash scripts/run_conver_ckpt2pb.sh path/to/dataset path/to/ckpt

# example:
# bash scripts/run_conver_ckpt2pb.sh /home/nankai/dataset/mmnet results/train/MMNetModel-1593500
```
sh文件后需要指定数据集位置和ckpt模型文件路径，转换的pb模型会保存在offline_infer文件夹中，model.pb为图文件，mmnet.pb为网络静态模型文件，最终使用mmnet.pb 其输出节点被构建为'output'.
我们转换好的PB模型在 https://mmnet-qer.obs.cn-north-4.myhuaweicloud.com/mmnet.pb 和 https://pan.baidu.com/s/16SRdFLI5HpEwMvV3ezICAw 
提取码：qwzg 


### 2、PB模型转OM模型
使用HECS上的Mind-Studio转换PB模型到OM模型, 选择算子融合，以加速静态模型的运行速度，其输入格式为NHWC, 输出节点为output，其中后处理为图像的步骤已经包含在模型文件中。
我们转换好的OM模型在链接：https://mmnet-qer.obs.cn-north-4.myhuaweicloud.com/mmnet_fuse.om 和https://pan.baidu.com/s/1c13T_KLDV1g6jw60hOmsYQ 
提取码：2wt8 

### 3、数据预处理
读取数据集中所有图片，对其进行预处理，默认保存在offline_infer的Bin/test文件夹中d的images和masks
```
bash scripts/run_datapre.sh path/to/dataset
#example:
# bash scripts/run_datapre.sh /home/nankai/dataset/mmnet 
```


### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/ccl/msame)

### 5、推理性能精度测试
#### 推理性能测试
使用如下命令进行性能测试：
```
./msame --model ./mmnet.om --output ./output/ --loop 100
```


#### 推理精度测试
使用OM模型推理结果，运行：
```
./msame --model ./offline_infer/mmnet.om --input ./offline_infer/Bin/test/images --output ./offline_infer/Bin/test/outputs
```
所有的输出会保存在```./offline_infer/Bin/test/output```目录下，
或者从 ```OBS://``` 下载并解压到该目录下

运行以下命令进行离线推理精度测试, 默认加载上面保存的默认目录中的output和masks.
```
python offline_infer/evaluate.py 
```
离线推理精度MAD=0.023，与在线模型精度一致
