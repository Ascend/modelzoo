

# MobileNetv2 Inference for Tensorflow 

This repository provides a script and recipe to Inference the MobileNetv2 model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd ModelZoo_MobileNetv2_TF_HARD/ModelZoo_MobileNetv2_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset

   http://image-net.org/challenges/LSVRC/2012/index#data

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=$ PYTHONPATH:${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  ```
  atc --model=mobilenet_v2_tf.pb --framework=3 --output=mobilenet_v2_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=mobilenet_v2_tf_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  ./benchmark_tf.sh --batchSize=1 --modelType=mobilenetv2 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=mobilenet_v2_tf_aipp.om --dataPath=image-1024 --trueValuePath=val_lable.txt
  ```



## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 5W images | 71.75 %/ 90.48% |

#### Inference performance results

|       model       | batch size | Inference performance |
| :---------------: | :--------: | :-------------------: |
| offline Inference |     1      |       505 img/s       |