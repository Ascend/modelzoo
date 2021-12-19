

# ResCNN Inference for Tensorflow 

This repository provides a script and recipe to Inference of the ResCNN model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/ResCNN_for_ACL/
```

### 2. Download and preprocess the dataset

1. Download the DIV2K dataset by yourself. 

2. Put 100 LR pictures to './DIV2K_test_100/' as test data.

3. Make directories for inference input and output:
```
cd scripts
mkdir input_bins
mkdir results
```
   Temporary bin files will be saved.


### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/ResCNN_for_ACL.zip)

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om
  Because of the whole test picture will be split to some different sizes,including 64 x 64, 32 x 64, 32 x 44, etc, here,we convert three om files:

  ```
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_64_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,64,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_44_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,44,3" --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh
  ```

## NPU Performance
### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | data  |    mean-PSNR    | mean-SSIM|
| :---------------: | :-------: | :-------------: |:-------------:|
| offline Inference | 100 images | 23.748 |0.747|


## Reference
[1] https://github.com/payne911/SR-ResCNN-Keras

