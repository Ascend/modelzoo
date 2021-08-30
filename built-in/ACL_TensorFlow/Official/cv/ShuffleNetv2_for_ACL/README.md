

# ShuffleNetv2 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the ShuffleNetv2 model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/ShuffleNetv2_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 Validation dataset by yourself which includes 50000 JPEGS.

2. Move **ILSVRC2012val** to **'scripts/'**
```
———scripts
     |————ILSVRC2012val
           |————ILSVRC2012_val_00000001.JPEG
           |————ILSVRC2012_val_00000002.JPEG
           |————ILSVRC2012_val_00000003.JPEG
           |————ILSVRC2012_val_00000004.JPEG
           |————...
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 imagenet_preprocessing.py --src_path ./ILSVRC2012val/
```
The jpeg pictures will be preprocessed to bin fils.

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

  Download pb model of shufflenetv2 which was trained by this repo: [Repo of train](https://github.com/Ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/shufflenetv2/shufflenetv2_tf_uestclzx)
  [Pb Download Link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com:443/007_inference_backup/shufflenetv2/shufflenetv2_tf_uestclzx/Offline_shufflenetv2_tf_uestclzx/test_pb_model/shufflenetv2.pb?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1657964102&Signature=gJf%2BttvtPhUThIAEolQSOgO7CrI%3D)

  ```
  atc --model=shufflenetv2.pb --framework=3 --output=shufflenetv2_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --out_nodes="classifier/BiasAdd:0" --log=info
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
| offline Inference | 50000 images | 61.4 %/ 82.9% |


## Reference
https://github.com/Ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/shufflenetv2/shufflenetv2_tf_uestclzx
