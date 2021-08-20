

# Resnet34 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the Resnet34 model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Resnet34_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Tiny-ImageNet-200 dataset by yourself. [Download Link](https://tiny-imagenet.herokuapp.com/)

2. Move **tiny-imagenet-200** to **'scripts/'**
```
———scripts
     |————tiny-imagenet-200
           |————test
           |————train
           |————val
           |————wnids.txt
           |————words.txt
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 imagenet_tiny_preprocessing.py
```
The jpegs pictures will be preprocessed to bin fils.

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
  atc --model=resnet34_tf.pb --framework=3 --output=resnet34_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="test_inputs:1,64,64,3" --log=info
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

|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 10000 images | 51.8 %/ 76.2% |


## Reference
https://github.com/taki0112/ResNet-Tensorflow
