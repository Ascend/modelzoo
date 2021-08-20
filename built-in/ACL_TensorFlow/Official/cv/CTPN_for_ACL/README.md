# CTPN inference for Tensorflow

This repository provides a script and recipe to Inference the

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/CTPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
   ```
   https://github.com/eragonruan/text-detection-ctpn
   ```

2. Executing the Preprocessing Script
   ```
   python3 scripts/ctpn_dataPrepare.py --image_path=../image --output_path=$datasets --crop_width=1072 --crop_height=608 --img_conf=./img_info
   
   ```
3. Installing the BBOX
   ```
   cd scripts/utils/bbox
   bash make.sh
   ```
4. Download gt file by yourself and put it in the test directory
   ```
   https://rrc.cvc.uab.es/?ch=2&com=mymenthods&task=1
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

  ```
  atc --model=model/ctpn_tf.pb --framework=3 --output=model/ctpn_model --input_shape=input_image:1,608,1072,3 --soc_version=Ascend710 --enable_scope_fusion_passes=ScopeDynamicRNNPass --enable_small_channel=1
  ```

- Build the program 

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/ctpn_model.om --dataPath=../../datasets/ --modelType=ctpn  --imgType=rgb 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 233 images |    74.73%       |    70.26%       |    72.43%       |

