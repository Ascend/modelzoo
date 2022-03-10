# FACENET inference for Tensorflow

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
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Facenet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the lfw dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 align/align_dataset_mtcnn.py $cur_dir/lfw $dataset
   python3 preprocess_data.py  Path_of_Data_after_face_alignment  Outpath_of_Data_after_face_alignment  --use_fixed_image_standardization --lfw_batch_size 1
   
   ```
 
### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip)

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
   atc --framework=3 --model=./model/facenet_tf.pb  --output=./model/facenet --soc_version=Ascend310 --insert_op_conf=./facenet_tensorflow.cfg --input_format=NHWC --input_shape=input:4,160,160,3
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  ./benchmark_tf.sh --batchSize=4 --modelPath=../../model/facenet.om --dataPath=./dataset_bin/ --modelType=facenet --imgType=rgb
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    Embeddings Accuracy    |
| :---------------: | :---------: | :---------: |
| offline Inference | 12000 images |   99.532%     |

