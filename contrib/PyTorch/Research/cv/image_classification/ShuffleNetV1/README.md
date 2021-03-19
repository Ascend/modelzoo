# ShuffleNetV1 (size=1.0x, group=3)

## ImageNet training with PyTorch

This implements training of ShuffleNetV1 on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet).

## ShuffleNetV1 Detail

Base version of the model from [the paper author's code on Github](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1).
The training script is adapted from [the ShuffleNetV2 script on Gitee](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch/Official/cv/image_classification/Shufflenetv2_for_PyTorch).

## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Training
一、训练流程：
        
单卡训练流程：

    1.安装环境，并进入scripts目录
    2.修改train_1p.sh中的参数"data"为数据集路径
    3.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    4.执行train_1p.sh开始训练


    
多卡训练流程

    1.安装环境，并进入scripts目录
    2.修改train_8p.sh中的参数"data"为数据集路径
    3.修改参数device_id_list（多卡训练所使用的device id列表），为训练配置device_id，例如device_id=0,1,2,3,4,5,6,7
    4.执行train_8p.sh开始训练	

    
二、测试结果
    
训练日志路径：训练日志会输出到标准输出，请在执行训练脚本时重定向到文件。例如：

        nohup ./train_8p.sh > log.txt &

训练模型：训练生成的模型默认会写入到和sh文件同一目录（及scripts）下。当训练正常结束时，checkpoint.pth.tar为最终结果。


## ShufflenetV1 training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
| 67.21    | 462       | 1       | 240      | O2       |
| 66.45    | 3956      | 8       | 240      | O2       |
