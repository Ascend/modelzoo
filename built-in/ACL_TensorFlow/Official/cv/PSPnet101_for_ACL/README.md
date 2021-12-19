# PSPNet101 inference for Tensorflow

This repository provides a script and recipe to Inference the PSPNet101 model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/PSPNet101_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
2. Executing the Preprocessing Script
   ```
   #without flip
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset
   ```

   ```
   #flip
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset --flipped_eval --flipped_output_path=$flipped_dataset   
   ```

 
### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/PSPnet101_for_ACL.zip)

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
  atc --model=model/PSPNet101.pb --framework=3 --output=model/pspnet101_1batch --soc_version=Ascend710 --input_shape=input_image:1,1024,2048,3 --enable_small_channel=1 --insert_op_conf=pspnet_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  without flip
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb
  ```

  ```
  flip
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb --flippedDataPath=../../flipped_datasets/ --flippedEval=1
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Without flip  Inference accuracy results

|       model       | **data**   |    mIoU    | 
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |    77%     | 


### flip  Inference accuracy results

|       model       | **data**   |    mIoU    |    
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |   77.24%   | 

