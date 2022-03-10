# FACE-RESNET50 inference for Tensorflow

This repository provides a script and recipe to Inference the face-resnet50 model.

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
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/Face_Resnet50_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the lfw dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 align/align_dataset_mtcnn_facereset.py $cur_dir/lfw $dataset
   python3 preprocess.py $cur_dir/config/basemodel.py Path_of_Data_after_processing
   
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- get base model
  ```
  https://github.com/seasonSH/DocFace
  
  ```
- ckpt to pb
  ```
  for example:
   python3.7 /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py --input_checkpoint=./faceres_ms/ckpt-320000 --output_graph=./model/face_resnet50_tf.pb --output_node_names="embeddings" --input_meta_graph=./faceres_ms/graph.meta --input_binary=true
  ```

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Face_Resnet50_for_ACL.zip)

  ```
  /usr/local/Ascend/atc/bin/atc --model ./model/face_resnet50_tf.pb   --framework=3  --output=face_resnet50 --input_shape="image_batch:1,112,96,3" --enable_small_channel=1 --soc_version=Ascend710
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/face_resnet50.om --dataPath=./dataset_bin --modelType=faceresnet50 --imgType=rgb
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    Embeddings Accuracy    |
| :---------------: | :---------: | :---------: |
| offline Inference | 13233 images |   90.52%     |



