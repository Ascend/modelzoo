## ResNet50执行指南

### 1.git路径

https://gitlab.huawei.com/pmail_turinghava/training_shop/tree/master/03-code

### 2.数据集路径

TF2.X的ResNet50 与 TF1.15的ResNet50使用数据集相同，不做重复归档

```sh
# 数据集路径
/autotest/CI_daily/Resnet50_TF_daily/data/resnet50/imagenet_TF
```

### 3.拉起脚本

```shell
cd $path/test

########1P_256bs_SGD########
###稳定性### 全量42个epochs
train_full_1P.sh
###冒烟看护### 2个epochs
train_performance_1P.sh

########8P_256bs_SGD########
###稳定性### 42个epochs 
train_full_8P_256bs_SGD.sh
###冒烟看护### 6个epochs
train_performance_8P_256bs_SGD.sh

########8P_312bs_LARS########
###稳定性### 39个epochs
train_full_8P_312bs_LARS.sh
###冒烟看护### 6个epochs  
train_performance_8P_312bs_LARS.sh

#16P 256batch 42个epochs
train_full_16p.sh
```

### 4.开放参数及执行命令

```shell
# npu_config
# 必选
--data_path		           source data of training

# 可选
--precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
--over_dump		           if or not over detection, default is False
--data_dump_flag		   data dump flag, default is 0
--data_dump_step		   data dump step, default is 10
--profiling		           if or not profiling for performance debug, default is False
--autotune				   whether to enable autotune, default is False		

#示例：
./train_full_1p.sh --precision_mode=allow_fp32_to_fp16 --data_path=/home/imagenet_TF --data_dump_flag=True --data_dump_step=1 
```

### 5.结果校验

```shell
# 吞吐量
Final Performance ms/step ：
#总耗时(s)
Final Training Duration sec :
#精度
Final train_accuracy is：
```



