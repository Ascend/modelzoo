# <font face="微软雅黑">
# XACL FMK

***
XACL FMK工具可用作模型推理工具使用，集成了run_acl_model和run_acl_device工具功能，工具支持的推理场景如下：
* [x] Device合设部署场景下的om模型推理。
* [ ] Device拉远部署场景下的om模型推理。
* [x] 指定Device场景推理。
* [x] 循环Loop场景推理。
* [x] 数据Dump和性能采集场景推理。
* [x] 动态分档和动态Shape场景。
***

## 内容列表
- [工具编译](#工具编译)
- [工具使用](#工具使用)
    - [工具帮助](#工具帮助)
    - [普通场景](#普通场景)
    - [指定Device场景](#指定Device场景)
    - [循环Loop场景](#循环Loop场景)
    - [数据Dump和性能采集场景推理](#数据Dump和性能采集场景推理)
    - [动态分档和动态Shape场景](#动态分档和动态Shape场景)
    - [拉远部署场景](#拉远部署场景)

## 工具编译
1. 确认run包是否安装，且安装路径是否与CMakeLists.txt文件中相同：
```Bash
if (NOT DEFINED ENV{RUN_PATH})
    set(RUN_PATH "/usr/local/Ascend")
else ()
    set(RUN_PATH $ENV{RUN_PATH})
endif ()

message(STATUS "RUN_PATH: ${RUN_PATH}")

```
2. 确认acllib包是否安装，且安装路径是否与CMakeLists.txt文件中相同：
```Bash
# Header path
include_directories(${RUN_PATH}/acllib/include inc)

# add host lib path
link_directories(${RUN_PATH}/acllib/lib64/stub/)

```
3. ./xacl_fmk.sh执行脚本编译
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./xacl_fmk.sh 
-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/g++
-- Check for working CXX compiler: /usr/bin/g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- RUN_PATH: /usr/local/Ascend
-- Configuring done
-- Generating done
-- Build files have been written to: /home/wang-bain/xacl_fmk/build/intermediates/host
Scanning dependencies of target xacl_fmk
[ 33%] Building CXX object CMakeFiles/xacl_fmk.dir/utils.cpp.o
[ 66%] Building CXX object CMakeFiles/xacl_fmk.dir/main.cpp.o
[100%] Linking CXX executable /home/wang-bain/xacl_fmk/out/xacl_fmk
[100%] Built target xacl_fmk

```

## 工具帮助
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk/out# ./xacl_fmk -h
2021-04-22 21:39:51.794 - I - [XACL]: Usage: ./run_acl_model [input parameters]
-m=model                 Required, om model file path
                         Relative and absolute paths are supported

-i=inputFiles            Optional, input bin files or input file directories, use commas (,) to separate multiple inputs
                         Relative and absolute paths are supported, set inputs to all zeros if not specified
-o=outputPath            Required, path of output files
                         Relative and absolute paths are supported
-d=dumpJson              Optional, Configuration file used to save operator input and output data
                         The default value is NULL, indicating that operator input and output data is not saved

-v=dynamicShape          Optional, Size of the dynamic shape
                         Use semicolon (;) to separate each input, use commas (,) to separate each dim
                         The default value is NULL, indicating that the dynamicShape function is disabled
                         Enter the actual shape size when the dynamicShape function is enabled
-w=dynamicSize           Optional, Size of the output memory
                         Use semicolon (;) to separate each output
                         The default value is NULL, indicating that the dynamicShape function is disabled
                         Enter the actual output size when the dynamicShape function is enabled
-x=imageRank             Optional, Size of the height and width rank, use commas (,) to separate
                         The default value is NULL, indicating that the image rank function is disabled
                         Enter the actual height and width size when the image rank function is enabled
-y=batchRank             Optional, Size of the batch rank, cannot be used with heightRank or widthRank
                         The default value is 0, indicating that the batch rank function is disabled
                         Enter the actual size of the batch when the batch rank function is enabled
-z=dimsRank              Optional, Size of the dims rank, use commas (,) to separate
                         The default value is NULL, indicating that the dims rank function is disabled
                         Enter the actual size of each dims when the dims rank function is enabled

-n=nodeId                Optional, ID of the NPU used for inference
                         The default value is 0, indicating that device 0 is used for inference
-l=loopNum               Optional, The number of inference times
                         The default value is 1, indicating that inference is performed once
-b=batchSize             Optional, Size of the static batch
                         The default value is 1, indicating that the static batch is 1
                         Static batch will be disabled when dynamic batch has been set

-r=remoteDevice          Optional, Whether the NPU is deployed remotely
                         The default value is 0, indicating that the NPU is co-deployed as 1951DC
                         The value 1 indicates that the NPU is deployed remotely as 1951MDC
-g=mergeInput            Optional, whether merge input by batch size, only take effect in directories input
                         The default value is false, in this case, each input must be saved in N batches
                         Otherwise, each input must be saved in 1 batch and will be merged to N batches automatically

-h=help                  Show this help message

```

## 工具使用
***
1. 编译完成后可执行文件为： ./out/xacl_fmk。
2. 针对不同环境，请source ./env/AscendXXX_env.ini。
3. 环境变量中请根据实际部署场景修改ASCEND_HOME和PYTHON_HOME默认路径。
***

### 普通场景
#### 执行命令，入参为仅包含-m， -i， -o和-b， -i指定BIN文件输入
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -b 8 -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om -i ./input_ids_00000.bin,./input_mask_00000.bin,./segment_ids_00000.bin -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000
2021-03-31 11:25:01.856 - I - [XACL]: Static batch size is: 8
2021-03-31 11:25:01.856 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om
2021-03-31 11:25:01.856 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000
2021-03-31 11:25:01.856 - I - [XACL]: Input type is bin file
2021-03-31 11:25:01.856 - I - [XACL]: The number of input files is: 3
2021-03-31 11:25:01.856 - I - [XACL]: Check whether the input files are empty
2021-03-31 11:25:01.856 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_ids/input_ids_00000.bin is checked
2021-03-31 11:25:01.856 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_mask/input_mask_00000.bin is checked
2021-03-31 11:25:01.856 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/segment_ids/segment_ids_00000.bin is checked
2021-03-31 11:25:01.856 - I - [XACL]: The number of checked input files is: 3
2021-03-31 11:25:01.867 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:25:02.080 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:25:02.081 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:25:02.081 - I - [XACL]: Init acl interface success
2021-03-31 11:25:02.399 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:25:02.399 - I - [XACL]: Load acl model interface success
2021-03-31 11:25:02.399 - I - [XACL]: Create description interface success
2021-03-31 11:25:02.402 - I - [XACL]: Create input data interface success
2021-03-31 11:25:02.402 - I - [XACL]: Create output data interface success
2021-03-31 11:25:02.632 - I - [XACL]: Run acl model success
2021-03-31 11:25:02.632 - I - [XACL]: Loop 000, start timestamp 1617161102402, end timestamp 1617161102633, cost time 230.38ms
2021-03-31 11:25:02.633 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:25:02.633 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:25:02.633 - I - [XACL]: Single sample average NPU inference time of 1 loops: 230.38 ms 34.72 fps
2021-03-31 11:25:02.633 - I - [XACL]: Destroy input data success
2021-03-31 11:25:02.633 - I - [XACL]: Destroy output data success
2021-03-31 11:25:02.666 - I - [XACL]: Unload acl model success
2021-03-31 11:25:04.392 - I - [XACL]: 1 samples average NPU inference time: 230.38 ms 34.72 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_01_000.bin

```

#### 执行命令，入参为仅包含-m， -i， -o和-b， -i指定目录输入
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -b 8 -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om -i ./input_ids,./input_mask,./segment_ids -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx
2021-03-31 11:25:37.158 - I - [XACL]: Static batch size is: 8
2021-03-31 11:25:37.158 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om
2021-03-31 11:25:37.158 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx
2021-03-31 11:25:37.158 - I - [XACL]: Input type is director
2021-03-31 11:25:37.169 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:25:37.378 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:25:37.378 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:25:37.708 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:25:37.708 - I - [XACL]: Load acl model interface success
2021-03-31 11:25:37.708 - I - [XACL]: Create description interface success
2021-03-31 11:25:37.708 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_ids/input_ids_00000.bin is checked
2021-03-31 11:25:37.708 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_mask/input_mask_00000.bin is checked
2021-03-31 11:25:37.708 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/segment_ids/segment_ids_00000.bin is checked
2021-03-31 11:25:37.708 - I - [XACL]: Init acl interface success
2021-03-31 11:25:37.711 - I - [XACL]: Create input data interface success
2021-03-31 11:25:37.711 - I - [XACL]: Create output data interface success
2021-03-31 11:25:37.941 - I - [XACL]: Run acl model success
2021-03-31 11:25:37.941 - I - [XACL]: Loop 000, start timestamp 1617161137711, end timestamp 1617161137942, cost time 230.58ms
2021-03-31 11:25:37.942 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:25:37.942 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:25:37.942 - I - [XACL]: Single sample average NPU inference time of 1 loops: 230.58 ms 34.69 fps
2021-03-31 11:25:37.942 - I - [XACL]: Destroy input data success
2021-03-31 11:25:37.943 - I - [XACL]: Destroy output data success
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_ids/input_ids_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_mask/input_mask_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/segment_ids/segment_ids_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: Init acl interface success
2021-03-31 11:25:37.943 - I - [XACL]: Create input data interface success
2021-03-31 11:25:37.943 - I - [XACL]: Create output data interface success
2021-03-31 11:25:38.175 - I - [XACL]: Run acl model success
2021-03-31 11:25:38.175 - I - [XACL]: Loop 000, start timestamp 1617161137944, end timestamp 1617161138175, cost time 231.08ms
2021-03-31 11:25:38.175 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:25:38.175 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:25:38.175 - I - [XACL]: Single sample average NPU inference time of 1 loops: 231.08 ms 34.62 fps
2021-03-31 11:25:38.176 - I - [XACL]: Destroy input data success
2021-03-31 11:25:38.176 - I - [XACL]: Destroy output data success
2021-03-31 11:25:38.204 - I - [XACL]: Unload acl model success
2021-03-31 11:25:39.397 - I - [XACL]: 2 samples average NPU inference time: 230.83 ms 34.66 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_01_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00001_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00001_output_01_000.bin

```

#### 执行命令，入参为仅包含-m， -i， -o， -b和-g， -i指定目录输入
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -b 8 -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om -i ./input_ids,./input_mask,./segment_ids -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx -g true
2021-03-31 11:25:37.158 - I - [XACL]: Static batch size is: 8
2021-03-31 11:25:37.158 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8.om
2021-03-31 11:25:37.158 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx
2021-03-31 11:25:37.158 - I - [XACL]: Input type is director
2021-03-31 11:25:37.160 - I - [XACL]: Merge input flag is: true
2021-03-31 11:25:37.169 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:25:37.378 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:25:37.378 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:25:37.708 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:25:37.708 - I - [XACL]: Load acl model interface success
2021-03-31 11:25:37.708 - I - [XACL]: Create description interface success
2021-03-31 11:25:37.708 - I - [XACL]: Create merged input path: /home/wang-bain/npu/predict_bert/data/SQuAD1.1/input_ids_batch_8
2021-03-31 11:25:37.708 - I - [XACL]: Create merged input path: /home/wang-bain/npu/predict_bert/data/SQuAD1.1/input_mask_batch_8
2021-03-31 11:25:37.708 - I - [XACL]: Create merged input path: /home/wang-bain/npu/predict_bert/data/SQuAD1.1/segment_ids_batch_8
2021-03-31 11:25:37.708 - I - [XACL]: Start to merge input: input_ids
2021-03-31 11:25:37.709 - I - [XACL]: Merge input: input_ids to /home/wang-bain/npu/predict_bert/data/SQuAD1.1/input_ids_batch_8 finished
2021-03-31 11:25:37.709 - I - [XACL]: Start to merge input: input_mask
2021-03-31 11:25:37.710 - I - [XACL]: Merged input: input_mask to /home/wang-bain/npu/predict_bert/data/SQuAD1.1/input_mask_batch_8 finished
2021-03-31 11:25:37.710 - I - [XACL]: Start to merge input: segment_ids
2021-03-31 11:25:37.711 - I - [XACL]: Merged input: segment_ids to /home/wang-bain/npu/predict_bert/data/SQuAD1.1/segment_ids_batch_8 finished
2021-03-31 11:25:37.712 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_ids_batch_8/input_ids_00000.bin is checked
2021-03-31 11:25:37.712 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_mask_batch_8/input_mask_00000.bin is checked
2021-03-31 11:25:37.712 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/segment_ids_batch_8/segment_ids_00000.bin is checked
2021-03-31 11:25:37.712 - I - [XACL]: Init acl interface success
2021-03-31 11:25:37.712 - I - [XACL]: Create input data interface success
2021-03-31 11:25:37.712 - I - [XACL]: Create output data interface success
2021-03-31 11:25:37.941 - I - [XACL]: Run acl model success
2021-03-31 11:25:37.941 - I - [XACL]: Loop 000, start timestamp 1617161137711, end timestamp 1617161137942, cost time 230.58ms
2021-03-31 11:25:37.942 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:25:37.942 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:25:37.942 - I - [XACL]: Single sample average NPU inference time of 1 loops: 230.58 ms 34.69 fps
2021-03-31 11:25:37.942 - I - [XACL]: Destroy input data success
2021-03-31 11:25:37.943 - I - [XACL]: Destroy output data success
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_ids_batch_8/input_ids_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/input_mask_batch_8/input_mask_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: The input file: /home/wang-bain/bert_predict/data/SQuAD1.1/segment_ids_batch_8/segment_ids_00001.bin is checked
2021-03-31 11:25:37.943 - I - [XACL]: Init acl interface success
2021-03-31 11:25:37.943 - I - [XACL]: Create input data interface success
2021-03-31 11:25:37.943 - I - [XACL]: Create output data interface success
2021-03-31 11:25:38.175 - I - [XACL]: Run acl model success
2021-03-31 11:25:38.175 - I - [XACL]: Loop 000, start timestamp 1617161137944, end timestamp 1617161138175, cost time 231.08ms
2021-03-31 11:25:38.175 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:25:38.175 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:25:38.175 - I - [XACL]: Single sample average NPU inference time of 1 loops: 231.08 ms 34.62 fps
2021-03-31 11:25:38.176 - I - [XACL]: Destroy input data success
2021-03-31 11:25:38.176 - I - [XACL]: Destroy output data success
2021-03-31 11:25:38.204 - I - [XACL]: Unload acl model success
2021-03-31 11:25:39.397 - I - [XACL]: 2 samples average NPU inference time: 230.83 ms 34.66 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00000_output_01_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00001_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:25 Bert_SQuAD1_1_for_TensorFlow_BatchSize_8_Idx_00001_output_01_000.bin

```

#### 执行命令，入参为仅包含-m， -o， 不指定-i和-b
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000
2021-03-31 11:26:04.088 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om
2021-03-31 11:26:04.088 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000
2021-03-31 11:26:04.088 - I - [XACL]: Input type is empty, create all zero inputs
2021-03-31 11:26:04.099 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:26:04.308 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:26:04.309 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:26:04.309 - I - [XACL]: Init acl interface success
2021-03-31 11:26:04.595 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:26:04.595 - I - [XACL]: Load acl model interface success
2021-03-31 11:26:04.595 - I - [XACL]: Create description interface success
2021-03-31 11:26:04.595 - I - [XACL]: Create input data interface success
2021-03-31 11:26:04.595 - I - [XACL]: Create output data interface success
2021-03-31 11:26:04.829 - I - [XACL]: Run acl model success
2021-03-31 11:26:04.829 - I - [XACL]: Loop 000, start timestamp 1617161164596, end timestamp 1617161164829, cost time 233.42ms
2021-03-31 11:26:04.829 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:26:04.830 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:26:04.830 - I - [XACL]: Single sample average NPU inference time of 1 loops: 233.42 ms 4.28 fps
2021-03-31 11:26:04.830 - I - [XACL]: Destroy input data success
2021-03-31 11:26:04.830 - I - [XACL]: Destroy output data success
2021-03-31 11:26:04.863 - I - [XACL]: Unload acl model success
2021-03-31 11:26:06.401 - I - [XACL]: 1 samples average NPU inference time: 233.42 ms 4.28 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:26 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:26 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_01_000.bin

```

### 指定Device场景
#### 执行命令，入参增加-n，不设置时默认为0。
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000 -n 0
2021-03-31 11:26:33.065 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om
2021-03-31 11:26:33.065 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000
2021-03-31 11:26:33.065 - I - [XACL]: Device id is: 0
2021-03-31 11:26:33.065 - I - [XACL]: Input type is empty, create all zero inputs
2021-03-31 11:26:33.076 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:26:33.287 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:26:33.288 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:26:33.288 - I - [XACL]: Init acl interface success
2021-03-31 11:26:33.605 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:26:33.605 - I - [XACL]: Load acl model interface success
2021-03-31 11:26:33.605 - I - [XACL]: Create description interface success
2021-03-31 11:26:33.606 - I - [XACL]: Create input data interface success
2021-03-31 11:26:33.606 - I - [XACL]: Create output data interface success
2021-03-31 11:26:33.839 - I - [XACL]: Run acl model success
2021-03-31 11:26:33.839 - I - [XACL]: Loop 000, start timestamp 1617161193606, end timestamp 1617161193840, cost time 233.41ms
2021-03-31 11:26:33.840 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:26:33.840 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:26:33.840 - I - [XACL]: Single sample average NPU inference time of 1 loops: 233.41 ms 4.28 fps
2021-03-31 11:26:33.840 - I - [XACL]: Destroy input data success
2021-03-31 11:26:33.840 - I - [XACL]: Destroy output data success
2021-03-31 11:26:33.873 - I - [XACL]: Unload acl model success
2021-03-31 11:26:35.405 - I - [XACL]: 1 samples average NPU inference time: 233.41 ms 4.28 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:26 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:26 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_01_000.bin

```

### 循环Loop场景
#### 执行命令，入参增加-l，不设置时默认为1。
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000 -n 0 -l 2
2021-03-31 11:27:04.414 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om
2021-03-31 11:27:04.414 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000
2021-03-31 11:27:04.414 - I - [XACL]: Device id is: 0
2021-03-31 11:27:04.414 - I - [XACL]: Loop numbers are: 2
2021-03-31 11:27:04.414 - I - [XACL]: Input type is empty, create all zero inputs
2021-03-31 11:27:04.425 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:27:04.659 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:27:04.660 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:27:04.660 - I - [XACL]: Init acl interface success
2021-03-31 11:27:04.980 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:27:04.980 - I - [XACL]: Load acl model interface success
2021-03-31 11:27:04.980 - I - [XACL]: Create description interface success
2021-03-31 11:27:04.980 - I - [XACL]: Create input data interface success
2021-03-31 11:27:04.980 - I - [XACL]: Create output data interface success
2021-03-31 11:27:05.214 - I - [XACL]: Run acl model success
2021-03-31 11:27:05.214 - I - [XACL]: Loop 000, start timestamp 1617161224981, end timestamp 1617161225214, cost time 233.39ms
2021-03-31 11:27:05.215 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:27:05.215 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:27:05.446 - I - [XACL]: Run acl model success
2021-03-31 11:27:05.446 - I - [XACL]: Loop 001, start timestamp 1617161225216, end timestamp 1617161225446, cost time 230.72ms
2021-03-31 11:27:05.446 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:27:05.447 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:27:05.447 - I - [XACL]: Single sample average NPU inference time of 2 loops: 232.05 ms 4.31 fps
2021-03-31 11:27:05.447 - I - [XACL]: Destroy input data success
2021-03-31 11:27:05.447 - I - [XACL]: Destroy output data success
2021-03-31 11:27:05.474 - I - [XACL]: Unload acl model success
2021-03-31 11:27:07.409 - I - [XACL]: 1 samples average NPU inference time: 232.05 ms 4.31 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/bert_predict/save/output# ls -l
-rw-r--r-- 1 root root 12288 Mar 31 11:27 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_00_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:27 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_00_001.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:27 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_01_000.bin
-rw-r--r-- 1 root root 12288 Mar 31 11:27 Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000_output_01_001.bin

```

### 数据Dump和性能采集场景推理
#### 获取模型json
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# atc --mode=1 --om=./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om --json=./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.json
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.

```
#### 获取模型名称
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# less ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.json | tail -n 3
  "name": "Bert_SQuAD1_1_for_TensorFlow_BatchSize_1",
  "version": 2
}

```
#### 创建acl.json
采集精度dump数据
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk/out# vim acl.json 
{
    "dump":{ 
        "dump_list": [{"model_name": "Bert_SQuAD1_1_for_TensorFlow_BatchSize_1"}],    # 上一步获取的模型名称
        "dump_path": "/home/wang-bain/xacl_fmk/out",                                  # Dump数据的目录
        "dump_mode": "all"                                                            # Dump数据方式，all表示dump输入输出，output表示仅dump输出（默认），input表示仅dump输入
    }       
}

```
采集性能profiling数据
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk/out# vim acl.json 
{
    "profiler":{ 
        "switch": "on",                                                               # 是否打开profiling采集开关
        "output": "/home/wang-bain/xacl_fmk/out",                                     # profiling采集文件输出目录，会在此目录下创建JOB目录
        "aicpu": "on",                                                                # 是否采集AICPU算子profiling数据
        "aic_metrics": "PipeUtilization"                                              # AICORE算子采集数据类型
    }       
}

```
**【注意】** 
1. acl.json文件格式务必保证和上述样例一致，同时采集精度和性能数据时，可以合并json，并确保**json格式合法**。
2. acl.json样例中的 # 注释仅作为样例，使用时一定要**删除**，json文件是**不支持**注释的。

#### 执行命令，入参增加-d，不设置时默认为空。
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om -o ./Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000 -d acl.json
2021-03-31 11:30:23.246 - I - [XACL]: Om model file is: /home/wang-bain/bert_predict/save/model/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1.om
2021-03-31 11:30:23.246 - I - [XACL]: Output file prefix is: /home/wang-bain/bert_predict/save/output/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1_Idx_00000
2021-03-31 11:30:23.246 - I - [XACL]: Dump json file is: acl.json
2021-03-31 11:30:23.246 - I - [XACL]: Input type is empty, create all zero inputs
2021-03-31 11:30:23.287 - I - [XACL]: Interface of aclInit return success
2021-03-31 11:30:23.529 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-31 11:30:23.530 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-31 11:30:23.530 - I - [XACL]: Init acl interface success
2021-03-31 11:30:23.891 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-31 11:30:23.891 - I - [XACL]: Load acl model interface success
2021-03-31 11:30:23.891 - I - [XACL]: Create description interface success
2021-03-31 11:30:23.892 - I - [XACL]: Create input data interface success
2021-03-31 11:30:23.892 - I - [XACL]: Create output data interface success
2021-03-31 11:30:41.936 - I - [XACL]: Run acl model success
2021-03-31 11:30:41.936 - I - [XACL]: Loop 000, start timestamp 1617161423892, end timestamp 1617161441936, cost time 18043.79ms
2021-03-31 11:30:42.190 - I - [XACL]: Dump output 00 data to file success
2021-03-31 11:30:42.190 - I - [XACL]: Dump output 01 data to file success
2021-03-31 11:30:42.190 - I - [XACL]: Single sample average NPU inference time of 1 loops: 18043.79 ms 0.06 fps
2021-03-31 11:30:42.190 - I - [XACL]: Destroy input data success
2021-03-31 11:30:42.190 - I - [XACL]: Destroy output data success
2021-03-31 11:30:42.234 - I - [XACL]: Unload acl model success
2021-03-31 11:30:43.939 - I - [XACL]: 1 samples average NPU inference time: 18043.79 ms 0.06 fps

```
#### 检查执行结果，dump数据以时间戳形式存放在acl.json指定目录下
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ll out/20210331113023/0/Bert_SQuAD1_1_for_TensorFlow_BatchSize_1/1/0/
-rw------- 1 root root 14745806 Mar 31 11:30 Add.bert_embeddings_addbert_embeddings_add_1.12.4.1617161424189416
-rw------- 1 root root 37748895 Mar 31 11:30 BatchMatMul.bert_encoder_layer_0_attention_self_MatMul_1.23.4.1617161425150706
...
-rw------- 1 root root    24674 Mar 31 11:30 TransposeD.transpose.198.4.1617161441769837
-rw------- 1 root root    24678 Mar 31 11:30 Unpack.unstack.199.4.1617161441772690

```

### 动态分档和动态Shape场景
#### 动态分档场景适用于动态HWSize，动态Batch和动态Dims三种场景
***
1. 使用-x HeightSize,WidthSize 表示动态HWSize场景下的HW实际大小，以逗号分隔，如： -x 128,128。
2. 使用-y BatchSize 表示动态Batch场景下的Batch实际大小，如： -y 32。
3. 使用-z Dim0Size,Dim1Size...,DimNSize 表示动态Dims场景下输入各维度实际大小，以逗号分隔各维度，如： -z 32,3,128,128。
***
#### 动态Shape场景
***
1. 使用-v Dim0Size,Dim1Size...,DimNSize 表示动态Shape场景下输入的各维度实际大小，以逗号分割各维度，以分号分割各输入，如：-v 32,3,128,128;32,3,128,128
2. 使用-w Memory 表示动态Shape场景下输出的内存大小，以分号分割各输出，如：-w 1024;2048
***

#### 此处以动态Batch分档为例
#### 模型转换时增加--dynamic_batch_size入参，并指定分档Batch大小
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk/out# atc --model=./support_reduce_nd_nd_001.pb --framework=3 --output=./support_reduce_nd_nd_001 --soc_version=AscendA310 --input_shape="Placeholder:-1,56,56,32" --dynamic_batch_size="1,2"
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.

```
#### 执行命令，入参增加-y，在模型转换中指定的分档Batch中取值，不设置时默认为0。当指定-y时，-b参数无效，静态Batch将重置为动态Batch取值
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m support_reduce_nd_nd_001.om -i support_reduce_nd_nd_001_input0.bin,support_reduce_nd_nd_001_input1.bin -o support_reduce_nd_nd_001_npu -y 2
2021-03-20 10:34:40.611 - I - [XACL]: Om model file is: support_reduce_nd_nd_001.om
2021-03-20 10:34:40.611 - I - [XACL]: The number of input files is: 2
2021-03-20 10:34:40.611 - I - [XACL]: Check whether the input files are empty
2021-03-20 10:34:40.611 - I - [XACL]: The input file: support_reduce_nd_nd_001_input0.bin is checked
2021-03-20 10:34:40.611 - I - [XACL]: The input file: support_reduce_nd_nd_001_input1.bin is checked
2021-03-20 10:34:40.611 - I - [XACL]: The number of checked input files is: 2
2021-03-20 10:34:40.611 - I - [XACL]: Output file prefix is: support_reduce_nd_nd_001_npu
2021-03-20 10:34:40.611 - I - [XACL]: Dynamic batch size is: 2
2021-03-20 10:34:40.621 - I - [XACL]: Interface of aclInit return success
2021-03-20 10:34:40.980 - I - [XACL]: Interface of aclrtSetDevice return success
2021-03-20 10:34:40.981 - I - [XACL]: Interface of aclrtCreateContext return success
2021-03-20 10:34:40.981 - I - [XACL]: Init acl interface success
2021-03-20 10:34:40.993 - I - [XACL]: Interface of aclmdlLoadFromFile return success
2021-03-20 10:34:40.993 - I - [XACL]: Load acl model interface success
2021-03-20 10:34:40.993 - I - [XACL]: Create description interface success
2021-03-20 10:34:40.995 - I - [XACL]: Create input data interface success
2021-03-20 10:34:40.995 - I - [XACL]: Create output data interface success
2021-03-20 10:34:40.996 - I - [XACL]: Run acl model success
2021-03-20 10:34:40.996 - I - [XACL]: Loop 000, start timestamp 1616207680996, end timestamp 1616207680996, cost time 0.74ms
2021-03-20 10:34:40.996 - I - [XACL]: Dump output data to file success
2021-03-20 10:34:40.996 - I - [XACL]: NN inference cost average time: 0.74 ms 1347.71 fps
2021-03-20 10:34:42.357 - I - [XACL]: Destroy input data success
2021-03-20 10:34:42.357 - I - [XACL]: Destroy output data success
2021-03-20 10:34:42.357 - I - [XACL]: Unload acl model success
2021-03-31 10:34:43.153 - I - [XACL]: 1 samples average NPU inference time: 18043.79 ms 0.06 fps

```
#### 检查执行结果
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk/out# ls -l
-rw-r--r-- 1 root root 401408 Mar 20 10:34 support_reduce_nd_nd_001_npu_output_0_0.bin

```

### 拉远部署场景
#### 执行命令，入参增加-r，不设置时默认为0。
```Bash
root@ubuntu-7131:/home/wang-bain/xacl_fmk# ./out/xacl_fmk -m support_reduce_nd_nd_001.om -i support_reduce_nd_nd_001_input0.bin -o support_reduce_nd_nd_001_npu -r 1

```
具体执行结果待验证

# </font>
