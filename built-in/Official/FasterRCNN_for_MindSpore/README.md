# FasterRcnn Example
 
## Description
 
FasterRcnn is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

## Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use coco2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```
        pip install Cython

        pip install pycocotools

        pip install mmcv
        ```
        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

        ```
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017
    
        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organizing the dataset infomation as coco dataset format is suggested.


## Example structure
 
```shell
.
└─FasterRcnn          
  ├─README.md    
  ├─scripts                   
    ├─run_standalone_train.sh               
	├─run_distribute_train.sh               
	└─run_eval.sh                          
  ├─src        
    ├─FasterRcnn            
	  ├─__init__.py            
	  ├─anchor_generator.py            
	  ├─bbox_assign_sample.py            
	  ├─bbox_assign_sample_stage2.py           
	  ├─faster_rcnn_r50.py            
	  ├─fpn_neck.py           
	  ├─proposal_generator.py            
	  ├─rcnn.py           
	  ├─resnet50.py            
	  ├─roi_align.py            
	  └─rpn.py       
	├─datasets        
	  ├─coco           
	    ├─coco_util.py    
	├─config.py   
	├─dataset.py        
	├─lr_schedule.py        
	├─network_define.py       
	└─util.py   
  ├─eval.py                                 
  └─train.py                                 
```

## Running the example

### Train
 
#### Usage

```
# distributed training
sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [PRETRAINED_MODEL]
 
# standalone training
sh run_standalone_train.sh [PRETRAINED_MODEL]
```
 
> Rank_table.json which is specified by MINDSPORE_HCCL_CONFIG_PATH is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

> As for PRETRAINED_MODEL，if not set, the model will be trained from the very beginning.

#### Result
 
Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the followings in loss.log.

 
```
# distribute training result(8p)
epoch: 1 step: 7393, rpn_loss: 0.12054, rcnn_loss: 0.40601, rpn_cls_loss: 0.04025, rpn_reg_loss: 0.08032, rcnn_cls_loss: 0.25854, rcnn_reg_loss: 0.14746, total_loss: 0.52655
epoch: 2 step: 7393, rpn_loss: 0.06561, rcnn_loss: 0.50293, rpn_cls_loss: 0.02587, rpn_reg_loss: 0.03967, rcnn_cls_loss: 0.35669, rcnn_reg_loss: 0.14624, total_loss: 0.56854
epoch: 3 step: 7393, rpn_loss: 0.06940, rcnn_loss: 0.49658, rpn_cls_loss: 0.03769, rpn_reg_loss: 0.03165, rcnn_cls_loss: 0.36353, rcnn_reg_loss: 0.13318, total_loss: 0.56598
...
epoch: 10 step: 7393, rpn_loss: 0.03555, rcnn_loss: 0.32666, rpn_cls_loss: 0.00697, rpn_reg_loss: 0.02859, rcnn_cls_loss: 0.16125, rcnn_reg_loss: 0.16541, total_loss: 0.36221
epoch: 11 step: 7393, rpn_loss: 0.19849, rcnn_loss: 0.47827, rpn_cls_loss: 0.11639, rpn_reg_loss: 0.08209, rcnn_cls_loss: 0.29712, rcnn_reg_loss: 0.18115, total_loss: 0.67676
epoch: 12 step: 7393, rpn_loss: 0.00691, rcnn_loss: 0.10168, rpn_cls_loss: 0.00529, rpn_reg_loss: 0.00162, rcnn_cls_loss: 0.05426, rcnn_reg_loss: 0.04745, total_loss: 0.10859
```

### Infer
 
#### Usage
 
```
# infer
sh run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```
 
> checkpoint can be produced in training process.

#### Result
 
Inference result will be stored in the example path, whose folder name is "infer". Under this, you can find result like the followings in log.
 
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```
