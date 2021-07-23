## ST_GCN模型推理指导：

### 文件说明
    1. st_gcn_export.py：       onnx模型导出脚本
    2. st_gcn_preprocess.py：   模型预处理脚本
    3. st_gcn_postprocess.sh：  模型后处理脚本
    4. st_gcn_atc.sh:           om模型转换脚本


### 环境依赖
    pip3 install torch==1.2.0
    pip3 install torchvision==0.4.0
    pip3 install numpy==1.19.2
    pip3 install onnx==1.7.0


### 环境准备
    下载ST_GCN源码或解压文件
    1. 下载链接：https://github.com/open-mmlab/mmskeleton
        下载源码包解压到指定目录下

    2. 获取权重文件
       https://download.openmmlab.com/mmskeleton/checkpoints/st_gcn_kinetics-6fa43f73.pth

    3. 将软件包中文件包括st_gcn_export.py、st_gcn_preprocess.py、st_gcn_postprocess.py等拷到mmskeleton当前目录下
   

### 推理端到端步骤

1. pth导出onnx
    ```
    python3.7 st_gcn_export.py -ckpt=./checkpoints/st_gcn_kinetics-6fa43f73.pth -onnx=./st-gcn_kinetics-skeleton_bs1.onnx -batch_size=1
    ```

2. 配置离线推理环境变量
    设置推理环境变量：
    ```
    export install_path=/usr/local/Ascend/ascend-toolkit/latest
    export PATH=/usr/local/python3.7/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
    export PYTHONPATH=/usr/local/python3.7/lib/python3.7/site-packages/:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg/auto_tune/:$PYTHONPATH
    export LD_LIBRARY_PATH=${install_path}/acllib/lib64/:${install_path}/atc/lib64:$LD_LIBRARY_PAT
    export ASCEND_OPP_PATH=${install_path}/opp/
    ```

3. 利用ATC工具转换为om
    ```
    bash st_gcn_atc.sh
    ```   

4. 模型前处理
    数据集预处理
    ```
    python3.7 st_gcn_preprocess.py -data_dir=./data/kinetics-skeleton/val_data.npy -label_dir=./data/kinetics-skeleton/val_label.pkl
    -output_path=./data/kinetics-skeleton/ -batch_size=1 -num_workers=1
    ```

    数据集创建
    ```
    python3.7 get_info.py bin ./data/kinetics-skeleton/val_data/ kinetics.info 300 18
    ```
    
5. benchmark推理
    ```
    ./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=3 -om_path=./st-gcn_bs1_sim.om -input_width=300 -input_height=8
    -input_text_path=./kinetics.info -useDvpp=false -output_binary=true
    ```

6. 模型后处理
    ```
    python3.7 st_gcn_postprocess.py -result_dir=./result/dumpOutput_device0/ -label_dir=./data/Kinetics/kinetics-skeleton/val_label.pkl
    ```