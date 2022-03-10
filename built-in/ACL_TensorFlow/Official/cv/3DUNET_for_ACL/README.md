# 3DUNET inference for Tensorflow

This repository provides a script and recipe to Inference the

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Research/cv/3DUNET_for_Tensorflow
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself

2. Put the dataset files to **'3DUNET_for_ACL/ori_images/'** like this:
```
--MICCAI_BraTS_2019_Data_Training

```

3. Executing the Preprocessing Script
   ```
   mkdir ori_images/tfrecord
   python3 scripts/preprocess_data.py --input_dir=ori_images/MICCAI_BraTS_2019_Data_Training/ --output_dir=ori_images/tfrecord
   python3 scripts/prepocess.py ./ori_images/tfrecord/ ./datasets ./labels
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

- convert pb to om(Ascend710)  
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend710 --input_shape=input:1,224,224,160,4 --enable_small_channel=1
  ```

- convert pb to om(Ascend310)
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend310 --input_shape=input:1,224,224,160,4 --optypelist_for_implmode=ReduceMeanD --op_select_implmode=high_precision
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

|       model       | **data**   |       TumorCore     | PeritumoralEdema | EnhancingTumor | MeanDice | WholeTumor |   
| :---------------: | :-------:  | :-----------------: |  :-------------: | :------------: |:--------:|:----------:|
| offline Inference |  68 images |        72.59%       |      78.48%      |     70.46%     |  73.84%  |   90.74%   |

