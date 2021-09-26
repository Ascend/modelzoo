

# AlignedReID Inference for Tensorflow 

This repository provides a script and recipe to Inference of the AlignedReID model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/AlignedReID_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the cuhk03 dataset by yourself.

2. Put whole dataset to **'scripts/data'**.

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

  ```
  atc --model=AlignedReID_tf.pb --framework=3 --output=AlignedReID_100batch --output_type=FP32 --soc_version=Ascend310 --input_shape="images_total:100,224,224,3" --log=info
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

|       model       | **data**  |    Mean-cmc    |
| :---------------: | :-------: | :-------------: |
| offline Inference | cuhk03 val | 99.4% |

## Reference

[1] https://github.com/Phoebe-star/AlignedReID