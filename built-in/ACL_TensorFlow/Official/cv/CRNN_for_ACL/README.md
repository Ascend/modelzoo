

# CRNN Inference for Tensorflow 

This repository provides a script and recipe to Inference the CRNN model. Original train implement please follow this link: [CRNN_for_Tensorflow](https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/detection/CRNN_for_TensorFlow)

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://github.com/Ascend/modelzoo.git
cd modelzoo/built-in/ACL/Official/cv/CRNN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the IIIT5K/ICDAR03/SVT test dataset by yourself and put them to the path: **scripts/data/**

2. Preprocess of the test datasets and labels:
```
cd scripts
python3 tools/preprocess.py
```
and it will generate **img_bin** and **labels** directories:
```
img_bin
|___batch_data_000.bin
|___bathc_data_001.bin
...

labels
|___batch_label_000.txt
|___batch_label_001.txt
...
```

### 3. Offline Inference
**Freeze ckpt to pb**

Please use the frozen_graph.py from the train scripts: [frozen_graph.py](https://github.com/Ascend/modelzoo/blob/master/built-in/TensorFlow/Official/cv/detection/CRNN_for_TensorFlow/tools/frozen_graph.py)
```
python3 frozen_graph.py --ckpt_path= ckpt_path/shadownet_xxx.ckpt-600000
```

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/CRNN_for_ACL.zip)

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  ```
  atc --model=shadownet_tf_64batch.pb --framework=3 --output=shadownet_tf_64batch --output_type=FP32 --soc_version=Ascend310 --input_shape="test_images:64,32,100,3" --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```
  An executable file **benchmark** will be generated under the path: **Benchmark/output/**

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results:

| Test Dataset | Per_Char_Accuracy | Full_Seq_Accuracy |
|--------------|-------------------|-------------------|
| SVT          | 88.9%             | 77.2%             |
| ICDAR2013    | 93.5%             | 87.3%             |
| IIIT5K       | 91.4%             | 79.6%             |
