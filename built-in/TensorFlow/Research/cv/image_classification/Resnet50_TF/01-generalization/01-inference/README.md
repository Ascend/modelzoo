
## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd ModelZoo_Resnet50_TF_Atlas
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset

    http://image-net.org/challenges/LSVRC/2012/index#data 

### 3. Convert ckpt to pb
- cd **00-ckpt2pb**

- **edit** export_pb_from_meta_npu.sh (see example below)

- bash *export_pb_from_meta_npu.sh*

- Examples:

    add your own configuration info in *export_pb_from_meta_npu.sh*. The default configuration info could be considered as an example.

    ```shell
    #!/bin/bash
    export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
    export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/code
    export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
    export ASCEND_OPP_PATH=/usr/local/Ascend/opp
    
    python3.7 export_pb_from_meta_npu.py ./mode/model.ckpt-450500 resnet_model/final_dense resnet50_npu.pb
    ```

    In *export_pb_from_meta_npu.py* , python scripts part have three args. 

    ```
    input_checkpoint=sys.argv[1]  # the path of checkpoint
    output_node_names = sys.argv[2] # output nodes name
    output_graph=sys.argv[3]  # the path of pb
    ```

    Run the program, and then you will get inception_v4_tf.pb in your set path.

    ```shell
    bash export_pb_from_meta_npu.sh
    ```



4. Online inference

- cd */Resnet50_TF/**01-onlineInfer**

- Run inference demo

  ``` 
  python3.7 resnet50_infer_tf.py \
            --batchsize=1 \
            --model_path=model/resnet50_910to310.pb \
            --image_path=image_50000/*.JPEG \
            --label_file=val_label.txt \
            --input_tensor_name='input_data:0' \
            --output_tensor_name='resnet_model/final_dense:0'
  ```



5. Offline inference

- cd */Resnet50_TF/**02-offlineInfer**

- Configure the env according to your install path

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=$ PYTHONPATH:${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- Convert pb to om

  ```
  atc --model=resnet50_910to310_tf.pb --framework=3 --output=resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32 --output_type=FP32 --soc_version=Ascend310 --input_shape="input_data:1,224,224,3" --insert_op_conf=test_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the proram

  ```
  cd scripts
  chmod +x benchmark_tf.sh
  ./benchmark_tf.sh --batchSize=1 --modelType=resnet50 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=../../model/resnet/resnet50_tf_aipp_quantized.om --dataPath=../../datasets/resnet/image-1024 --trueValuePath=../../datasets/resnet/input_50000.csv
  ```

  

## Performance## 

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

####  Inference accuracy results

| model             | **data**  | Top1/Top5     |
| ----------------- | --------- | ------------- |
| online Inference  | 5W images | 76.4 %/94.0 % |
| offline Inference | 5W images | 75.2 %/92.6 % |

####  Inference performance results

| model             | batch size | Inference performance |
| ----------------- | ---------- | --------------------- |
| online Inference  | 1          | 534 img/s             |
| offline Inference | 1          | 378img/s              |