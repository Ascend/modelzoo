# Densenet24 inference for Tensorflow

This repository provides a script and recipe to Inference the

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Densenet24_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
2. Put the dataset files to **'Densenet24_for_ACL/ori_images'** like this:
```
--ori_images
  |----BRATS2017
     |----Brats17ValidationData
       |----Brats17_2013_21_1
          |----xxxxx.nii.gz
       |----Brats17_2013_25_1
       |----Brats17_2013_26_1
       |----Brats17_CBICA_ABM_1
       |----Brats17_CBICA_AUR_1
       |----Brats17_CBICA_AXN_1
       |----Brats17_CBICA_AXQ_1
       |----Brats17_TCIA_192_1
       |----Brats17_TCIA_319_1
       |----Brats17_TCIA_377_1
       |----val.txt
       |----val.txt.zl
  |----npu
     |----d24_correction-4.index
     |----d24_correction-4.meta
```
3. Executing the Preprocessing Script
   ```
   cd scripts
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

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/DenseNet24_for_ACL.zip)

  ```
  cd ..
  atc --model=model/densenet24.pb --framework=3 --output=model/densenet24_1batch --soc_version=Ascend710 --input_shape="Placeholder:1,38,38,38,2;Placeholder_1:1,38,38,38,2"
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

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       |  **data**  |   TumorCore   |    PeritumoralEdema    |    EnhancingTumor   |
| :---------------: |  :------:  | :-----------: | :--------------------: | :-----------------: |
| offline Inference |  10 images |     99.588%   |         99.812%        |        99.901%      |

