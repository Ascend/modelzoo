# 目标检测与目标分割类模型使用说明

## Requirements
* NPU配套的run包安装(C20B030)
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)


## Dataset Prepare
1. 下载COCO数据集，放在datasets中。如已有下载可通过设置环境变量DETECTRON2_DATASETS=“coco所在数据集路径”进行设置，如export DETECTRON2_DATASETS=/home/sample，则coco数据集放在/home/sample目录中
## 1P
1. 编辑并运行run.sh
以下是mask_rcnn的训练脚本
```
source ./env_b031.sh   #设置环境变量
export PYTHONPATH=./:$PYTHONPATH
export SLOG_PRINT_TO_STDOUT=0
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 0"  #0-3的device日志格式设置为error级别
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(2)[disable]\" "

nohup python3.7 tools/train_net.py \     # 运行入口文件
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \  # 选择mask_rcnn的模型yaml文件
        --debug-mode 0\  # 代码运行日志级别
        AMP 1\ #是否使用apex优化训练
        OPT_LEVEL O2 \ # apex优化级别
        LOSS_SCALE_VALUE 64 \ 
        MODEL.DEVICE npu:0 \  # 指定对应的device_id
        SOLVER.IMS_PER_BATCH 2 \ # 设置batch_size
        SOLVER.MAX_ITER 84000 \ # 设置训练iter数
        SEED 1234 \ # 随机数种子
        MODEL.RPN.NMS_THRESH 0.8 \ # NMS 阈值
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \ # proposal采样率
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \ # 图片读取进程数
        SOLVER.BASE_LR 0.0025 & # 初始学习率 

```
若将以上的--config-file参数指定为：configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml，则可训练对应的faster_rcnn模型。其他参数含义一致。

## 4P
1. 编辑并运行run4p.sh

```
source ./env_b031.sh
export PYTHONPATH=./:$PYTHONPATH
export SLOG_PRINT_TO_STDOUT=0
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 0"
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(2)[disable]\" "

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        --num-gpus 4\
        --debug-mode 0\
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        SOLVER.IMS_PER_BATCH 8 \
        SOLVER.MAX_ITER 61000 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \
        SOLVER.BASE_LR 0.01 &

```
若将以上的--config-file参数指定为：configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml，则可训练对应的faster_rcnn模型。其他参数含义一致。

## Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch_maskrcnn.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch_maskrcnn.tar pytorch:b020_maskrcnn
2.执行docker_start.sh 后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020_maskrcnn /train/coco /home/MaskRCNN
