# Video Restoration with Enhanced Deformable Convolution Networks

## 简介

这是EDVR的一份tensorflow复现代码，支持在NPU、GPU和CPU上进行EDVR的训练和推理。关于EDVR模型本身，请参考[此论文](https://arxiv.org/abs/1905.02716)。

## 环境需求

- tensorflow==1.15
- imageio
- yacs
- python3.7
- scikit-image (可选)

## 工程目录结构

```sh
edvr
|-- ascendcv
|   |-- layers
|   |-- runner
|   `-- utils
|-- ascendvsr
|   |-- config
|   |-- layers
|   `-- models
|-- configs
|   `-- edvr.yaml
|-- data
|   `-- reds
|       |-- images
|       |   `-- 001
|       |       |-- blur4
|       |       `-- truth
|       `-- sets
|-- scripts
|   |-- 2p.json
|   |-- 8p.json
|   |-- download_REDS.py
|   |-- make_reds_dataset.py
|   |-- prepare_8p.sh
|   |-- prepare_reds_dataset.sh
|   |-- regroup_reds_dataset.py
|   |-- run_1p_train
|   |-- run_1p_train.sh
|   |-- run_8p_train.sh
|   |-- run_evaluation.sh
|   |-- run_freeze.sh
|   `-- run_inference.sh
`-- tools
    `-- main.py
```

- ascendcv: 一些基础的层，和运行态所需的类
- ascendvsr: 一些基础的模块，视频超分基类，以及视频超分模型
- configs: 配置文件样例
- data: 数据集路径
- scripts: 顶层调用的shell文件，训练、推理等
- tools: 入口python脚本

## 准备数据集

以REDS4数据集为例

1. 运行如下脚本下载REDS4数据对应的4个部分

    ```sh
    # 当前目录: path/to/edvr
    datadir=./data/reds
    
    mkdir -p ${datadir}
    python3 scripts/download_REDS.py --root_dir ${datadir} --train_sharp --train_sharp_bicubic --val_sharp --val_sharp_bicubic
    ```

    这一步会将这四个部分下载下来，保存到``data/reds``目录中

2. 参考论文方法，合并原始训练集和验证集

    ```sh
    python3 scripts/regroup_reds_dataset.py ${datadir}
    ```

    这一步会将验证集（``val_sharp``和``val_sharp_bicubic``）合并到对应的初始训练集中。验证集部分将被重命名为序列240-269，并附在原训练集之后。因此总共有270个视频序列，每个视频序列有100帧。其中000，011，015，020四个序列将作为真正的验证集。

3. 重组数据集，与训练脚本进行适配

    ```sh
    mkdir ${datadir}/images
    
    mv ${datadir}/train_sharp ${datadir}/images/truth
    mv ${datadir}/train_sharp_bicubic/X4 ${datadir}/images/blur4
    ```

4. 准备数据meta信息

    ```sh
    mkdir ${datadir}/sets
    python3 scripts/make_reds_dataset.py ${datadir}/sets
    ```

我们已将上述4个步骤合并成一个shell脚本。用户可直接运行该脚本（仅适用于REDS4数据集）：

```sh
bash scripts/prepare_reds_dataset.sh
```

注意，第一步的数据下载可能会花费较多时间。

### 数据集目录结构

```sh
data/reds
|-- images
|   |-- 000
|   |   |-- blur4
|   |   |   |-- 00000000.png
|   |   |   |-- 00000001.png
|   |   |   |-- 00000002.png
|   |   |   |-- 00000003.png
|   |   |   |-- ...
|   |   |   `-- 00000099.png
|   |   `-- truth
|   |       |-- 00000000.png
|   |       |-- 00000001.png
|   |       |-- 00000002.png
|   |       |-- 00000003.png
|   |       |-- ...
|   |       `-- 00000099.png
|   |-- 001
|   |   |-- blur4
|   |   `-- truth
|   |-- 002
|   |   |-- blur4
|   |   `-- truth
|   |-- 003
|   |-- ...
|   |-- ...
|   `-- 269
`-- sets
    |-- train.json
    `-- val.json
```

### 自定义数据

建议用户按照上述目录结构构建数据集，方便适配脚本

## 训练

1. 在0卡上训练EDVR模型： 

    ```sh
    bash scripts/run_1p_train.sh 0 1
    ```

    第一个输入参数0表示设备ID，8卡环境一般为0-7；第二个参数是所使用的设备数量。

    用户可使用``yaml``配置文件来配置训练的部分超参、数据集的处理过程等信息。可配置项请参考``ascendvsr/config/defaults.py``文件，示例配置可参考``configs/edvr.yaml``。

    训练过程和状态将被打印在屏幕上，如loss值，迭代次数，迭代时间，数据帧率。打屏间隔由``cfg.solver.print_interval``进行设置，训练中间结果每隔一段迭代次数就会被保存到``cfg.output_dir``目录下（默认设置为``outputs/edvr``） ，保存间隔由 ``cfg.solver.checkpoint_interval`` 进行确定。 

    示例训练过程（batchsize=4）：

    > 2020-12-25 08:34:46 Step:20, lr:0.00040000, loss:9476.21899544, time:209.62ms, fps:19.08
    > 2020-12-25 08:34:51 Step:40, lr:0.00040000, loss:9483.95068774, time:210.05ms, fps:19.04
    > 2020-12-25 08:34:55 Step:60, lr:0.00040000, loss:9374.16324098, time:212.36ms, fps:18.84
    > 2020-12-25 08:34:59 Step:80, lr:0.00040000, loss:9282.14592199, time:215.24ms, fps:18.58
    > 2020-12-25 08:35:03 Step:100, lr:0.00040000, loss:9225.38266487, time:214.05ms, fps:18.69
    > 2020-12-25 08:35:08 Step:120, lr:0.00040000, loss:9147.80390486, time:216.20ms, fps:18.50
    > 2020-12-25 08:35:12 Step:140, lr:0.00040000, loss:9152.80633528, time:210.89ms, fps:18.97
    > 2020-12-25 08:35:16 Step:160, lr:0.00040000, loss:9033.11667964, time:221.91ms, fps:18.03
    > 2020-12-25 08:35:21 Step:180, lr:0.00040000, loss:8864.82630793, time:214.61ms, fps:18.64
    > 2020-12-25 08:35:25 Step:200, lr:0.00040000, loss:8713.98834194, time:213.67ms, fps:18.72
    > 2020-12-25 08:35:29 Step:220, lr:0.00040000, loss:8513.21058127, time:211.83ms, fps:18.88
    > 2020-12-25 08:35:33 Step:240, lr:0.00040000, loss:8290.43024285, time:212.71ms, fps:18.81
    > 2020-12-25 08:35:38 Step:260, lr:0.00040000, loss:8128.87668816, time:212.40ms, fps:18.83
    > 2020-12-25 08:35:42 Step:280, lr:0.00040000, loss:7993.71964797, time:215.15ms, fps:18.59
    > 2020-12-25 08:35:46 Step:300, lr:0.00040000, loss:7808.95153805, time:214.33ms, fps:18.66
    > 2020-12-25 08:35:50 Step:320, lr:0.00040000, loss:7670.07353983, time:215.52ms, fps:18.56
    > 2020-12-25 08:35:55 Step:340, lr:0.00039999, loss:7567.70286385, time:213.47ms, fps:18.74
    > 2020-12-25 08:35:59 Step:360, lr:0.00039999, loss:7432.47448890, time:209.44ms, fps:19.10
    > 2020-12-25 08:36:03 Step:380, lr:0.00039999, loss:7331.30056232, time:214.45ms, fps:18.65
    > 2020-12-25 08:36:07 Step:400, lr:0.00039999, loss:7172.49776584, time:212.36ms, fps:18.84
    > 2020-12-25 08:36:12 Step:420, lr:0.00039999, loss:7051.55225264, time:209.99ms, fps:19.05
    > 2020-12-25 08:36:16 Step:440, lr:0.00039999, loss:6938.33076964, time:214.65ms, fps:18.64
    > 2020-12-25 08:36:20 Step:460, lr:0.00039999, loss:6866.46246160, time:212.75ms, fps:18.80
    > 2020-12-25 08:36:24 Step:480, lr:0.00039999, loss:6809.04133592, time:212.99ms, fps:18.78
    > 2020-12-25 08:36:29 Step:500, lr:0.00039999, loss:6736.72430321, time:210.67ms, fps:18.99

2. 执行单机8卡训练：

    ```sh
    bash scripts/prepare_8p.sh
    bash scripts/run_8p_train.sh
    ```

    示例训练过程（每张卡上batchsize=4，设置只打屏device 0的训练结果）：

    > 2021-01-05 14:05:28 Step:20, lr:0.00040000, loss:10128.23235736, time:225.60ms, fps:141.85
    > 2021-01-05 14:05:32 Step:40, lr:0.00040000, loss:9911.20655139, time:230.63ms, fps:138.75
    > 2021-01-05 14:05:37 Step:60, lr:0.00040000, loss:9746.84064846, time:225.71ms, fps:141.78
    > 2021-01-05 14:05:41 Step:80, lr:0.00040000, loss:9610.47396385, time:225.90ms, fps:141.66
    > 2021-01-05 14:05:46 Step:100, lr:0.00040000, loss:9442.95130887, time:223.77ms, fps:143.00
    > 2021-01-05 14:05:50 Step:120, lr:0.00040000, loss:9300.51361234, time:228.85ms, fps:139.83
    > 2021-01-05 14:05:55 Step:140, lr:0.00040000, loss:9029.22531498, time:227.36ms, fps:140.75
    > 2021-01-05 14:05:59 Step:160, lr:0.00040000, loss:8850.09716859, time:225.94ms, fps:141.63
    > 2021-01-05 14:06:04 Step:180, lr:0.00040000, loss:8620.74724067, time:228.48ms, fps:140.05
    > 2021-01-05 14:06:09 Step:200, lr:0.00040000, loss:8384.87165179, time:227.64ms, fps:140.57
    > 2021-01-05 14:06:13 Step:220, lr:0.00040000, loss:8193.46840854, time:230.90ms, fps:138.59
    > 2021-01-05 14:06:18 Step:240, lr:0.00040000, loss:8052.94862988, time:229.94ms, fps:139.17
    > 2021-01-05 14:06:22 Step:260, lr:0.00040000, loss:7884.65315172, time:226.39ms, fps:141.35
    > 2021-01-05 14:06:27 Step:280, lr:0.00040000, loss:7719.58562702, time:225.88ms, fps:141.67
    > 2021-01-05 14:06:31 Step:300, lr:0.00040000, loss:7546.77749729, time:229.57ms, fps:139.39
    > 2021-01-05 14:06:36 Step:320, lr:0.00040000, loss:7387.72234128, time:228.84ms, fps:139.83
    > 2021-01-05 14:06:40 Step:340, lr:0.00039999, loss:7278.32803386, time:229.31ms, fps:139.55
    > 2021-01-05 14:06:45 Step:360, lr:0.00039999, loss:7133.84139243, time:226.94ms, fps:141.01
    > 2021-01-05 14:06:50 Step:380, lr:0.00039999, loss:7018.06870953, time:227.69ms, fps:140.54
    > 2021-01-05 14:06:54 Step:400, lr:0.00039999, loss:6941.95861095, time:228.10ms, fps:140.29
    > 2021-01-05 14:06:59 Step:420, lr:0.00039999, loss:6833.76219723, time:225.16ms, fps:142.12
    > 2021-01-05 14:07:03 Step:440, lr:0.00039999, loss:6742.74311336, time:226.48ms, fps:141.29
    > 2021-01-05 14:07:08 Step:460, lr:0.00039999, loss:6676.80909872, time:227.10ms, fps:140.91
    > 2021-01-05 14:07:12 Step:480, lr:0.00039999, loss:6571.46454924, time:226.12ms, fps:141.52
    > 2021-01-05 14:07:17 Step:500, lr:0.00039999, loss:6452.07785241, time:223.62ms, fps:143.10

## 精度验证

对训练模型进行精度验证的过程非常简单：

```sh
bash scripts/run_evaluation.sh 0 outputs/edvr/EDVR-600000
```

其中，第一个参数是设备ID，第二个参数是模型ckpt名。执行这一脚本后打屏输出类似于：

```sh
Evaluate 000
Video 000 PSNR = 27.58365249633789
        Inference time: 102.73
Evaluate 011
Video 011 PSNR = 30.992197036743164
        Inference time: 102.67
Evaluate 015
Video 015 PSNR = 33.03461837768555
        Inference time: 102.22
Evaluate 020
Video 020 PSNR = 29.36363983154297
        Inference time: 102.83
PSNR = 30.243528366088867
```

## 推理

推理调用接口与验证过程类似：

```sh
bash scripts/run_inference.sh 0 outputs/edvr/EDVR-600000
```

推理结果（超分图像）会保存在``${cfg.output_dir}/test`` 文件夹中。

## 固化

执行脚本：

```sh
bash scripts/run_freeze.sh outputs/edvr/EDVR-600000
```

固化的pb文件会被保存为``${cfg.output_dir}/EDVR.pb``。

如果固化时希望``batchsize``维度为``None``，以便推理时可以泛化到任意batchsize的场景，可以在固化时设置``data.eval_batch_size=-1``，脚本
将自动调整batchsize维度为``None``；此外，如果需要固化pb文件支持4D输入``[N*D,H,W,C]``，而非5D输入``[N,D,H,W,C]``（此处``D``是EDVR
输入连续帧数，区别于batchsize），可以在固化时选择设置``model.input_format_dimension=4``(默认为5)，脚本将在模型搭建时自动插入reshape
算子进行转换，保证网络后续结构正确。模型默认输出数据类型为``tf.float32``，如需调整pb模型的输出节点数据类型为uint8，在固化时选择设置
``model.convert_output_to_uint8=True``（默认为False）即可。

## 复现精度与性能

在REDS4数据集上

|      | 训练输入尺寸（每张卡） | 训练耗时 (ms/step) | 推理输入尺寸 (单卡) | 推理耗时 (ms/step) | PSNR (dB)   |
| ---- | ---------------------- | ------------------ | ------------------- | ------------------ | ----------- |
| 1p   | [32, 5, 64, 64, 3]     | 875                | [1, 5, 180, 320, 3] | 102                | 30.24352837 |
| 8p   | [4, 5, 64, 64, 3]      | 230                | [1, 5, 180, 320, 3] | 102                | 30.24139595 |

## 注意事项

1. 如需在GPU上进行训练，可将``cfg.device``设为``GPU``，其他可保持不变。需要注意的是，GPU版本暂只能在单卡上执行
2. （**重要**）独立的*deformable convolution*算子在C76B220版本及其之后才有。更早的版本只能使用tf算子组装起来的算子，在NPU上性能较差。配置项``cfg.edvr.impl``可选择使用NPU独立算子（如果可以的话）还是tf组装算子。如果``cfg.edvr.impl=tf``使用tf组装算子，而``cfg.edvr.impl='npu'`` 将使用NPU独立算子。

