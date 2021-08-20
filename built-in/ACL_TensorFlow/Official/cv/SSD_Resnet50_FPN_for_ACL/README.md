# SSD-RESNET50FPN inference for Tensorflow

This repository provides a script and recipe to Inference the

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/SSD_Resnet50_FPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the coco2014 dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 scripts/ssd_dataPrepare.py --input_file_path=Path of the image --output_file_path=Binary path for inference --crop_width=Width of the image cropping --crop_height=height of the image cropping --save_conf_path=Image configuration file path
   
   ```
3. Download gt labels
   [instances_minival2014.json](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/ID1654_ssd_resnet50fpn/scripts/instances_minival2014.json?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656057065&Signature=ydPmdux71bGzs38Q/xV7USQIdCg%3D)

   put json file to **'scripts'**
 
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
  atc --model=model/ssd-resnet50fpn_tf.pb --framework=3 --output=model/ssd_resnet50_fpn --output_type=FP16 --soc_version=Ascend710 --input_shape="image_tensor:1,640,640,3" "input_name1:image_tensor" --enable_scope_fusion_passes=ScopeBatchMultiClassNMSPass,ScopeDecodeBboxV2Pass,ScopeNormalizeBBoxPass,ScopeToAbsoluteBBoxPass
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/ssd_resnet50_fpn.om --dataPath=../../datasets/ --modelType=ssd_resnet50_fpn --imgType=rgb --trueValuePath=../../scripts/instances_minival2014.json
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    map      |
| :---------------: | :---------: | :---------: |
| offline Inference | 4952 images |   37.8%     |

