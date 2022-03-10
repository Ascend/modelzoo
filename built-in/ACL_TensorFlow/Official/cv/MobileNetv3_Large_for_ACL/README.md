# MOBILENETV3LARGE inference for Tensorflow

This repository provides a script and recipe to Inference the

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/MobileNetv3_Large_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 scripts/mobilenet_data_prepare.py --image_path=Path of the dataset --out_path=Dataset output path
   
   ```
 
### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/MobileNetv3_Large_for_ACL.zip)

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  ```
  atc --model=model/mobilenetv3large_tf.pb --framework=3 --output=model/mobilenetv3_large_aipp --output_type=FP32 --insert_op_conf=./mobilenetv3_tensorflow.cfg --input_shape=input:1,224,224,3 --soc_version=Ascend710 --fusion_switch_file=./mobilenetv3_fusion_config.json
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1  --outputType=fp32 --modelPath=../../model/mobilenetv3_large_aipp.om --dataPath=../../datasets/imagenet_50000/ --modelType=mobilenetv3_large --imgType=rgb --trueValuePath=../../datasets/input_5w.csv
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 50K images | 75.7 %/ 92.8%   |

