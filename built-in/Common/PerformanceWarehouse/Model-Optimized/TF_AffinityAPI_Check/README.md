# 性能检查分析工具

## 功能概述

### 简述

该工具通过扫描GE图、profiling数据、网络代码和log日志，发现可能存在的性能提升点，给出相对的建议。

### 检查点分类

工具主要针对以下几类进行扫描:

- 是否开启```allow_mix_precision```，并使能```Npu LossScale```
- 是否存在可以替换为Huawei 亲和API的接口(gelu、LSTM、dropout等)
- 数据预处理是否使用Dataset队列模式，并进行下沉处理
- 是否存在数据类型为INT64的AICPU算子

## 工具使用

### 工具获取

1. 下载压缩包的方式获取: 将https://github.com/Ascend/modelzoo 以压缩包的形式下载
2. 使用git命令进行获取
3. 移动 ```built-in/Common/PerformanceWarehouse/Model-Optimized/TF_AffinityAPI_Check``` 子目录至训练工作目录

### 安装第三方依赖

```shell
pip3 install pandas 
```

### 工具执行依赖及获取方法

- NPU的计算图

```shell
export PRINT_MODEL=1
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=./GE_DUMP_DIR  # 自行指定
```

- profiling数据

```shell
export PROFILING_MODE=True
export PROFILING_OPTIONS='{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","aicpu":"on","aic_metrics":"PipeUtilization","fp_point":"","bp_point":""}'
```

- log日志

```shell
export ASCEND_GLOBAL_LOG_LEVEL=1
```

### 执行命令

```shell
python3 main.py --tfadpater_graph_dir=${GE_DUMP_DIR} --profiling_data_dir=${PROFILING_DIR} --training_code_dir=${CODE_DIR} --npu_log_dir=${LOG_DIR} --net_name=${NAME}

--tfadpater_graph_dir  NPU计算图生成路径
--profiling_data_dir   profiling数据生成路径,下面包含JOB*文件夹
--training_code_dir    网络脚本路径
--npu_log_dir          NPU日志保存路径
--net_name			   网络名字，如ResNet50
```

## 结果查看

- excel文件

命中的点会生成在名为 ```analysis.xlsx```  的excel文件中, 用户可针对相应的建议做出修改，检查性能是否有提升。

- log日志

每一条检查点会生成在  ```suggestions.log``` 文件中。

