# <font face="微软雅黑">

# OpenPose Inference for TensorFlow
This repository provides a script and recipe to Inference the OpenPose model.

## Quick Start Guide
### 1. Clone the respository
```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/OpenPose_for_ACL
```

### 2. Download and preprocess the dataset

Download the COCO2014 dataset by yourself, more details see: [dataset](./dataset/coco/README.md)


### 3. Obtain the pb model

Obtain the OpenPose pb model, more details see: [models](./models/README.md)

### 4. Obtain process scripts

Obtain pafprocess and slidingwindow packages from: [tf_openpose](https://github.com/BoomFan/openpose-tf/tree/master/tf_pose) and put them into libs


### 5. Offline Inference
**Preprocess the dataset**
```Bash
python3 preprocess.py \
    --resize 656x368 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --output-dir ../input/

```

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

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/OpenPose_for_ACL.zip)

  ```
  atc --framework=3 \
      --model=./models/OpenPose_for_TensorFlow_BatchSize_1.pb \
      --output=./models/OpenPose_for_TensorFlow_BatchSize_1 \
      --soc_version=Ascend310 \
      --input_shape="image:1,368,656,3"
  ```

**Build the program**
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)

**Run the inference**
```
/xacl_fmk -m ./models/OpenPose_for_TensorFlow_BatchSize_1.om \
    -o ./output/openpose \
    -i ./input \
    -b 1
```

**PostProcess**
```
python3 postprocess.py \
    --resize 656x368 \
    --resize-out-ratio 8.0 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --data-idx 100 \
    --output-dir ../output/openpose 
```

**Sample scripts**
We also supoort the predict_openpose.sh to run the steps all above except **build the program**

### 6.Result
***
OpenPose Inference ：

| Type | IoU | Area | MaxDets | Result |
| :------- | :------- | :------- | :------- | :------- |
| Average Precision  (AP) | 0.50:0.95 | all | 20 | 0.399 |
| Average Precision  (AP) | 0.50 | all | 20 | 0.648 |
| Average Precision  (AP) | 0.75| all | 20 | 0.400 |
| Average Precision  (AP) | 0.50:0.95 | medium | 20 | 0.364 |
| Average Precision  (AP) | 0.50:0.95 | large | 20 | 0.443 |
| Average Recall     (AR) | 0.50:0.95 | all | 20 | 0.456 |
| Average Recall     (AR) | 0.50 | all | 20 | 0.683 |
| Average Recall     (AR) | 0.75 | all | 20 | 0.465 |
| Average Recall     (AR) | 0.50:0.95 | medium | 20 | 0.371 |
| Average Recall     (AR) | 0.50:0.95 | large | 20 | 0.547 |

***

## Reference

[1] https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess/


# </font>