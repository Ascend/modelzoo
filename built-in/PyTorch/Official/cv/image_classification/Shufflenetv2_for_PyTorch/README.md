# ShuffleNetV2

## ImageNet training with PyTorch

This implements training of ShuffleNetV2 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## ShuffleNetV2 Detail

Base version of the model from [shufflenetv2.py](https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py)
As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, ShufflenetV2 is re-implemented using semantics such as custom OP. For details, see models/shufflenetv2_wock_op_woct.py .

git 
## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 
    一、训练流程：
        
    单卡训练流程：
    
    ```
        1.安装环境
        2.修改run_1p.sh字段"data"为当前磁盘的数据集路径
        3.修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
        4.cd到run_1p.sh文件的目录，执行bash run_1p.sh单卡脚本， 进行单卡训练
    ```
    
        
    多卡训练流程
    
    ```
        1.安装环境
        2.修改多P脚本中字段"data"为当前磁盘的数据集路径
        3.修改字段device_id_list（多卡训练所使用的device id列表），为训练配置device_id，比如4p,device_id_list=0,1,2,3；8P默认使用0，1，2，3，4，5，6，7卡不用配置
        4.cd到run_8p.sh文件的目录，执行bash run_8p.sh等多卡脚本， 进行多卡训练	
    ```
        
    二、Docker容器训练：
        
    1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:
    
    ```
            docker import ubuntuarmpytorch.tar pytorch:b020
    ```
    2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：
    
    ```
            ./docker_start.sh pytorch:b020 /train/imagenet /home/Shufflenetv2_for_PyTorch
    ```
    3.执行步骤一训练流程（环境安装除外）
        
    三、测试结果
      
    训练日志路径：在训练脚本的同目录下result文件夹里，如：
    ```
            /home/Shufflenetv2_for_Pytorch/result/training_8p_job_20201121023601
    ```

## ShufflenetV2 training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
| 61.5     | 1200      | 1       | 20       | O2       |
| 68.5     | 2200      | 1       | 240      | O2       |
| 66.3     | 14000     | 8       | 240      | O2       |

- The 8p training precision is the same as that of the GPU. Compared with the 1p training precision, this is mainly caused by the large batch size (8192). You can use a distributed optimizer such as LARS to resolve this problem.