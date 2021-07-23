

# RetinaNet Inference for Tensorflow 

This repository provides a script and recipe to Inference of the RetinaNet model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Official/cv/RetinaNet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Voc2017 validation dataset by yourself. 

2. Put pictures to **'scripts/voc2017val'**

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 preprocess.py ./voc2017val/ ./input_bins/
```
   The pictures will be preprocessed to bin files and a **rawScale.txt** file will be created under **retinanet_postprocess** directory.

4. Split ground-truth labels from **instances_val2017.json**
```
python3 load_coco_json.py
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
  atc --model=retinanet_tf.pb --framework=3 --output=retinanet_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input_1:1,768,1024,3" --log=info
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
| offline Inference | 5000 images | 35.67% |

#### Inference performance results

|       model       | batch size | Inference performance |Platform |
| :---------------: | :--------: | :-------------------: |:-------------------: 
| offline Inference |     1      |       4.4 imgs/s       |  Ascend310*1        |
