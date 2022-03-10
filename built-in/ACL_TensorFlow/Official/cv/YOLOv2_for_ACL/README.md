
# YOLOv2 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the YOLOv2 model.

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
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/YOLOv2_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the VOC2007 test dataset by yourself, then extract **VOCtest_06-Nov-2007.tar**. 

2. Move VOC2007 test dataset to **'scripts/VOC2007'** like this:
```
VOC2007
|----Annotations
|----ImageSets
|----JPEGImages
|----SegmentationClass
|----SegmentationObject
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 preprocess.py ./VOC2007/JPEGImages/ ./input_bins/
```
   The pictures will be preprocessed to bin files.


4. Convert Groundtruth labels to text format
```
python3 xml2txt.py ./VOC2007/Annotations/ ./yolov2_postprocess/groundtruths/
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
  
  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/yolov2_tf.pb)

  以batcsize=1为例：

  ```
  atc --model=./yolov2.pb --input_shape='Placeholder:1,416,416,3'  --input_format=NHWC --output=./yolov2_tf_1batch --soc_version=Ascend310 --framework=3
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

|       model       | **data**  |    mAP    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 4952 images | 59.43% |

## Reference
[1] https://github.com/KOD-Chen/YOLOv2-Tensorflow


