# 2D_Attention_Unet inference for Tensorflow

This repository provides a script and recipe to offline inference

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
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/2D_Attention_Unet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Massachusetts Roads Dataset by yourself


2. Put the dataset files to **'2D_Attention_Unet_for_ACL/image_ori/'** like this:
```
--image_ori
  |----lashan
    |----test(xxx.tiff,total:49images)
    |----test_labels
    |----val
    |----val_labels
  |----Val(xxx_gt/xxx_img/xxx_pred .png)
```


3. Executing the Preprocessing Script
   ```
   cd scripts
   python3 preprocessdata_test.py --dataset=../image_ori/lashan --crop_height=224 --crop_width=224
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om


  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/2D_Attention_Unet_for_ACL.zip)

  ```
  cd ..
  mkdir model
  atc --model=model/2D_Attention_Unet_tf.pb --framework=3 --output=model/2DAttention_fp16_1batch --soc_version=Ascend710 --input_shape=inputs:1,224,224,3 --enable_small_channel=1 --insert_op_conf=2DAttention_aipp.cfg
  ```


- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**   |       accuracy      |    Road      |    Others    |    precision    |    F1_score    |    Iou    |
| :---------------: | :-------:  | :-----------------: | :----------: | :----------: | :-------------: | :------------: | :-------: |
| offline Inference |  49 images |     97.19%          |    60.25%    |    99.36%    |     97.88%      |      97.44%    |    76.02% |
