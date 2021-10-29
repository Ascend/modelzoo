# Densenet24 inference for Tensorflow

This repository provides a script and recipe to Inference the

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Densenet24_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
2. Executing the Preprocessing Script
   ```
   python3 preprocess.py -m ../ori_images/npu/dense24_correction-4 -mn dense24 -nc True -r ../ori_images/BRATS2017/Brats17ValidationData/ -input1 ../datasets/input_flair/ -input2 ../datasets/input_t1/
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
  atc --model=model/densenet24.pb --framework=3 --output=model/densenet24_1batch --soc_version=Ascend710 --input_shape="Placeholder:1,38,38,38,2;Placeholder_1:1,38,38,38,2"
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/densenet24_1batch.om --dataPath=../../datasets/input_flair/,../../datasets/input_t1/ --modelType=ID0121  --imgType=rgb 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       |  **data**  |   TumorCore   |    PeritumoralEdema    |    EnhancingTumor   |
| :---------------: |  :------:  | :-----------: | :--------------------: | :-----------------: |
| offline Inference |  10 images |     99.588%   |         99.812%        |        99.901%      |

