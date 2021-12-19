

# PixelLink Inference for Tensorflow 

This repository provides a script and recipe to Inference of the PixelLink model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/PixelLink_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Icdar2015 test dataset by yourself. You can get the test pictures(500 JPEGS)

2. Put JPEGS to **'scripts/ch4_test_images'**

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 pixellink_preprocessing.py ./ch4_test_images/ ./input_bins/
```
The jpegs pictures will be preprocessed to bin fils.

### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/PixelLink_for_ACL.zip)

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
  atc --model=pixellink_tf.pb --framework=3 --output=pixellink_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,768,1280,3" --insert_op_conf=pixellink_tf_aipp.json --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Hmean    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 500 images | 82.4% |

