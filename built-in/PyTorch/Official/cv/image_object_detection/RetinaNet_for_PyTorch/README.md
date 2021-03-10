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
```

#### Modified MMCV
将mmcv_need目录下的文件替换到mmcv的安装目录下。


### Build MMDET from source
1. 下载modelzoo项目zip文件并解压
2. 压缩modelzoo\built-in\PyTorch\Official\cv\image_object_detection\RetinaNet_for_PyTorch目录
3. 于npu服务器解压RetinaNet_for_PyTorch压缩包
4. 执行以下命令，安装mmdet
```
cd RetinaNet_for_PyTorch
pip install -r requirements/build.txt
pip install -v -e .
```


## Train MODEL

### 单P
1. 运行train_1p_cmd.sh
```
sh train_1p_cmd.sh
```

## 8P
1. 运行train_8p.sh
```
sh train_1p_cmd.sh
```

