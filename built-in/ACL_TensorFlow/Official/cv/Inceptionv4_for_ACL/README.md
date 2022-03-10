

# InceptionV4 Inference for Tensorflow 

This repository provides a script and recipe to Inference the InceptionV4 model.

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
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Inceptionv4_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset by yourself

 

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  #Please modify the environment settings as needed
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Inceptionv4_for_ACL.zip)

  For Ascend310:
  ```
  atc --model=inception_v4_tf.pb --framework=3 --output=inception_v4_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,299,299,3" --log=info --insert_op_conf=inception_v4_tf_aipp.cfg
  ```
  For Ascend710:
  ```
  atc --model=inception_v4_tf.pb --framework=3 --output=inception_v4_tf_aipp --output_type=FP32 --soc_version=Ascend710 --input_shape="input:1,299,299,3" --log=info --insert_op_conf=inception_v4_tf_aipp.cfg
  ```

- Build the program

  For Ascend310:
  ```
  unset ASCEND710_DVPP
  bash build.sh
  ```
  For Ascend710:
  ```
  export ASCEND710_DVPP=1
  bash build.sh
  ```
  
- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=inceptionv4 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=inception_v4_tf_aipp.om --dataPath=image-1024 --trueValuePath=val_lable.txt
  ```

## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model     |  SOC  | **data**  |    Top1/Top5    |
| :---------------:|:-------:|:-------: | :-------------: |
| offline Inference| Ascend310     | 50K images | 74.6 %/ 91.9% |
| offline Inference| Ascend710     | 50K images | 74.9 %/ 92.2% |

