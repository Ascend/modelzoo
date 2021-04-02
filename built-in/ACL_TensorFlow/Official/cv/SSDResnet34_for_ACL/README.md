

# GoogleNet Inference for Tensorflow 

This repository provides a script and recipe to Inference the GoogleNet model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/modelzoo.git
cd modelzoo/built-in/ACL/Official/cv/SSDResnet34_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the COCO2017 dataset by yourself

 

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  ```
  atc --model=ssdresnet34_1batch_tf.pb --framework=3 --output=ssdresnet34_1batch_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,300,300,3" --log=info --insert_op_conf=ssdresnet34_tf_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=ssdresnet34 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=ssdresnet34_1batch_tf_aipp.om --dataPath=COCO2017/val2017 --trueValuePath=instances_val2017.json
  ```



## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |  MAP  |
| :---------------: | :-------: | :---: |
| offline Inference | 5K images | 24.6% |

#### Inference performance results

|       model       | batch size | Inference performance |
| :---------------: | :--------: | :-------------------: |
| offline Inference |     1      |       139 img/s       |