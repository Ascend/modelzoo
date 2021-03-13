# Retinanet模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)


## Dataset Prepare
1. 下载COCO数据集
2. 新建文件夹data
3. 将coco数据集放于data目录下

### Build MMCV

#### MMCV full version with CPU
```
cd ../
git clone https://github.com/open-mmlab/mmcv.git

export MMCV_WITH_OPS=1
export MAX_JOBS=8

cd mmcv
python setup.py build_ext
python setup.py develop
pip list | grep mmcv
```

#### Modified MMCV
将mmcv_need目录下的文件替换到mmcv的安装目录下。
安装完mmdet后执行以下命令：
```
/bin/cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
/bin/cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
/bin/cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
```


### Build MMDET from source
1. 下载modelzoo项目zip文件并解压
2. 压缩modelzoo\built-in\PyTorch\Official\cv\image_object_detection\RetinaNet_for_PyTorch目录
3. 于npu服务器解压RetinaNet_for_PyTorch压缩包
4. 执行以下命令，安装mmdet
```
cd RetinaNet_for_PyTorch
pip install -r requirements/build.txt
pip install -v -e .
pip list | grep mm
```


## Train MODEL

### 导入环境变量
```
source pt_set_env.sh
```

### 单卡
1. 运行 train_retinanet_1p.sh
```
chmod +x ./tools/dist_train.sh
sh train_retinanet_1p.sh
```

### 8卡
1. 运行 train_retinanet_8p.sh
```
chmod +x ./tools/dist_train.sh
sh train_retinanet_8p.sh
```
