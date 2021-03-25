

# CRNN Inference for Tensorflow 

This repository provides a script and recipe to Inference the CRNN model. Original train implement please follow this link: [CRNN_for_Tensorflow](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/detection/CRNN_for_TensorFlow)

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/modelzoo.git
cd modelzoo/built-in/ACL/Official/cv/CRNN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the IIIT5K/ICDAR03/SVT test dataset by yourself and put them to the path: **scripts/data/**

2. Preprocess of the test datasets and labels:
```
cd scripts
python3 tools/preprocess.py
```
and it will generate **img_bin** and **labels** directories:
```
img_bin
|___batch_data_000.bin
|___bathc_data_001.bin
...

labels
|___batch_label_000.txt
|___batch_label_001.txt
...
```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  ```

- convert pb to om

  ```
  atc --model=alexnet_tf.pb --framework=3 --output=alexnet_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=alexnet_tf_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=alexnet --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=alexnet_tf_aipp.om --dataPath=image-1024 --trueValuePath=val_lable.txt
  ```



## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 5W images | 59.08 %/ 81.37% |

#### Inference performance results

|       model       | batch size | Inference performance |
| :---------------: | :--------: | :-------------------: |
| offline Inference |     1      |       270 img/s       |
