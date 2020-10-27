# Contents

- [DeepLabV3 Description](#DeepLabV3-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)


# [DeepLabV3 Description](#contents)
## Description
DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two keypoints of DeepLabV3: Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple scales, and augmented ASPP makes image-level features available to capture long range information. 
This repository provides a script and recipe to DeepLabV3 model and achieve state-of-the-art performance.

Refer to [this paper][1] for network details.
`Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.`

[1]: https://arxiv.org/abs/1706.05587

# [Model Architecture](#contents)
Resnet101 as backbone, atrous convolution for dense feature extraction.

# [Dataset](#contents)
Pascal VOC datasets and Semantic Boundaries Dataset
   - Download segmentation dataset.

   - Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

     ```
     JPEGImages/00001.jpg SegmentationClassGray/00001.png
     JPEGImages/00002.jpg SegmentationClassGray/00002.png
     JPEGImages/00003.jpg SegmentationClassGray/00003.png
     JPEGImages/00004.jpg SegmentationClassGray/00004.png
     ......
     ```

   - Configure and run build_data.sh to convert dataset to mindrecords. Arguments in scripts/build_data.sh:

     ```
     --data_root                 root path of training data
     --data_lst                  list of training data(prepared above)
     --dst_path                  where mindrecords are saved
     --num_shards                number of shards of the mindrecords
     --shuffle                   shuffle or not
     ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
  - Prepare hardware environment with Ascend. If you want to try Ascend , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)
- Install python packages in requirements.txt
- Generate config json file for 8pcs training
  
     ```
     # From the root of this project
     cd src/tools/
     python3 get_multicards_json.py 10.111.*.*
     # 10.111.*.* is the computer's ip address.
     ```

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:
- Runing on Ascend

Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is: 
```
run_standalone_train.sh
```
For 8 devices training, training steps are as follows:
1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is: 
```
run_distribute_train_s16_r1.sh
```
2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is:
```
run_distribute_train_s8_r1.sh
```
3. Train s8 with voctrain dataset, finetuning from model in pervious step, training script is:
```
run_distribute_train_s8_r2.sh
```
For evaluation, evaluating steps are as follows:
1. Eval s16 with voc val dataset, eval script is:
```
run_eval_s16.sh
```
2. Eval s8 with voc val dataset, eval script is:
```
run_eval_s8.sh
```
3. Eval s8 multiscale with voc val dataset, eval script is:
```
run_eval_s8_multiscale.sh
```
4. Eval s8 multiscale and flip with voc val dataset, eval script is:
```
run_eval_s8_multiscale_flip.sh
```

# [Script Description](#contents)
## [Script and Sample Code](#contents)
```shell
.
└──deeplabv3
  ├── README.md
  ├── script
    ├── build_data.sh                             # convert raw data to mindrecord dataset
    ├── run_distribute_train_s16_r1.sh            # launch ascend distributed training(8 pcs) with vocaug dataset in s16 structure
    ├── run_distribute_train_s8_r1.sh             # launch ascend distributed training(8 pcs) with vocaug dataset in s8 structure
    ├── run_distribute_train_s8_r2.sh             # launch ascend distributed training(8 pcs) with voctrain dataset in s8 structure
    ├── run_eval_s16.sh                           # launch ascend evaluation in s16 structure
    ├── run_eval_s8.sh                            # launch ascend evaluation in s8 structure
    ├── run_eval_s8_multiscale.sh                 # launch ascend evaluation with multiscale in s8 structure
    ├── run_eval_s8_multiscale_filp.sh            # launch ascend evaluation with multiscale and filp in s8 structure
    ├── run_standalone_train.sh                   # launch ascend standalone training(1 pc)
  ├── src
    ├── data                          
        ├── dataset.py                            # mindrecord data generator
        ├── build_seg_data.py                     # data preprocessing
    ├── loss
       ├── loss.py                                # loss definition for deeplabv3        
    ├── nets 
       ├── deeplab_v3
          ├── deeplab_v3.py                       # DeepLabV3 network structure
       ├── net_factory.py                         # set S16 and S8 structures
    ├── tools                    
       ├── get_multicards_json.py                 # get rank table file
    └── utils 
       └── learning_rates.py                      # generate learning rate                  
  ├── eval.py                                     # eval net
  ├── train.py                                    # train net
  └── requirements.txt                            # requirements file
```

## [Script Parameters](#contents)

Default Configuration
```
"data_file":"/PATH/TO/MINDRECORD_NAME"            # dataset path
"train_epochs":300                                # total epochs
"batch_size":32                                   # batch size of input tensor
"crop_size":513                                   # crop size
"base_lr":0.08                                    # initial learning rate
"lr_type":cos                                     # decay mode for generating learning rate
"min_scale":0.5                                   # minimum scale of data argumentation
"max_scale":2.0                                   # maximum scale of data argumentation
"ignore_label":255                                # ignore label
"num_classes":21                                  # number of classes
"model":deeplab_v3_s16                            # select model
"ckpt_pre_trained":"/PATH/TO/PRETRAIN_MODEL"      # path to load pretrain checkpoint
"is_distributed":                                 # distributed training, it will be True if the parameter is set
"save_steps":410                                  # steps interval for saving
"freeze_bn":                                      # freeze_bn, it will be True if the parameter is set
"keep_checkpoint_max":200                         # max checkpoint for saving
```

## [Training Process](#contents)

### Usage
#### Running on Ascend
Based on original DeepLabV3 paper, we reproduce two training experiments on vocaug (also as trainaug) dataset and evaluate on voc val dataset.

For single device training, please config parameters, training script is as follows: 
```
# run_standalone_train.sh
python ${train_code_path}/train.py --data_file=/PATH/TO/MINDRECORD_NAME  \
                    --train_dir=${train_path}/ckpt  \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
```
For 8 devices training, training steps are as follows:

1. Train s16 with vocaug dataset, finetuning from resnet101 pretrained model, script is as follows: 

```
# run_distribute_train_s16_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=32  \
                                               --crop_size=513  \
                                               --base_lr=0.08  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s16  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=410  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
2. Train s8 with vocaug dataset, finetuning from model in previous step, training script is as follows:
```
# run_distribute_train_s8_r1.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=800  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.02  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=820  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
3. Train s8 with voctrain dataset, finetuning from model in pervious step, training script is as follows:
```
# run_distribute_train_s8_r2.sh
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=$i
    export DEVICE_ID=`expr $i + $RANK_START_ID` 
    echo 'start rank='$i', device id='$DEVICE_ID'...'
    mkdir ${train_path}/device$DEVICE_ID
    cd ${train_path}/device$DEVICE_ID
    python ${train_code_path}/train.py --train_dir=${train_path}/ckpt  \
                                               --data_file=/PATH/TO/MINDRECORD_NAME  \
                                               --train_epochs=300  \
                                               --batch_size=16  \
                                               --crop_size=513  \
                                               --base_lr=0.008  \
                                               --lr_type=cos  \
                                               --min_scale=0.5  \
                                               --max_scale=2.0  \
                                               --ignore_label=255  \
                                               --num_classes=21  \
                                               --model=deeplab_v3_s8  \
                                               --loss_scale=2048  \
                                               --ckpt_pre_trained=/PATH/TO/PRETRAIN_MODEL  \
                                               --is_distributed  \
                                               --save_steps=110  \
                                               --keep_checkpoint_max=200 >log 2>&1 &
done
```
### Result

- Training vocaug in s16 structure
```
# distribute training result(8p)
epoch: 1 step: 41, loss is 0.8319108
Epoch time: 213856.477, per step time: 5216.012
epoch: 2 step: 41, loss is 0.46052963
Epoch time: 21233.183, per step time: 517.883
epoch: 3 step: 41, loss is 0.45012417
Epoch time: 21231.951, per step time: 517.852
epoch: 4 step: 41, loss is 0.30687785
Epoch time: 21199.911, per step time: 517.071
epoch: 5 step: 41, loss is 0.22769661
Epoch time: 21240.281, per step time: 518.056
epoch: 6 step: 41, loss is 0.25470978
...
```

- Training vocaug in s8 structure
```
# distribute training result(8p)
epoch: 1 step: 82, loss is 0.024167
Epoch time: 322663.456, per step time: 3934.920
epoch: 2 step: 82, loss is 0.019832281
Epoch time: 43107.238, per step time: 525.698
epoch: 3 step: 82, loss is 0.021008959
Epoch time: 43109.519, per step time: 525.726
epoch: 4 step: 82, loss is 0.01912349
Epoch time: 43177.287, per step time: 526.552
epoch: 5 step: 82, loss is 0.022886964
Epoch time: 43095.915, per step time: 525.560
epoch: 6 step: 82, loss is 0.018708453
Epoch time: 43107.458, per step time: 525.701
...
```

- Training voctrain in s8 structure
```
# distribute training result(8p)
epoch: 1 step: 11, loss is 0.00554624
Epoch time: 199412.913, per step time: 18128.447
epoch: 2 step: 11, loss is 0.007181881
Epoch time: 6119.375, per step time: 556.307
epoch: 3 step: 11, loss is 0.004980865
Epoch time: 5996.978, per step time: 545.180
epoch: 4 step: 11, loss is 0.0047651967
Epoch time: 5987.412, per step time: 544.310
epoch: 5 step: 11, loss is 0.006262637
Epoch time: 5956.682, per step time: 541.517
epoch: 6 step: 11, loss is 0.0060750707
Epoch time: 5962.164, per step time: 542.015
...
```

## [Evaluation Process](#contents)
### Usage
#### Running on Ascend
Config checkpoint with --ckpt_path, run script, mIOU with print in eval_path/eval_log.
```
./run_eval_s16.sh                     # test s16
./run_eval_s8.sh                      # test s8
./run_eval_s8_multiscale.sh           # test s8 + multiscale
./run_eval_s8_multiscale_flip.sh      # test s8 + multiscale + flip
```
Example of test script is as follows:
```
python ${train_code_path}/eval.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=16  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s8  \
                    --scales=0.5  \
                    --scales=0.75  \
                    --scales=1.0  \
                    --scales=1.25  \
                    --scales=1.75  \
                    --flip  \
                    --freeze_bn  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL >${eval_path}/eval_log 2>&1 &
```

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy
| **Network**    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| :----------: | :-----: | :----: | :----: | :-----: | :-----: | :-------------: |
| deeplab_v3 | √     |      |      |       | 77.37 | 77.21    |
| deeplab_v3 |       | √    |      |       | 78.84 | 78.51    |
| deeplab_v3 |       | √    | √    |       | 79.70 |79.45   |
| deeplab_v3 |       | √    | √    | √     | 79.89 | 79.77        |

Note: There OS is output stride, and MS is multiscale.

# [Model Description](#contents)
## [Performance](#contents)

### Evaluation Performance 
| Parameters                 | Ascend 910                                                   
| -------------------------- | -------------------------------------- |
| Model Version              | DeepLabV3                                       
| Resource                   | Ascend 910 | 
| Uploaded Date              | 09/04/2020 (month/day/year)          |             
| MindSpore Version          | 0.7.0-alpha                       |        
| Dataset                    | PASCAL VOC2012 + SBD              |       
| Training Parameters        | epoch = 300, batch_size = 32 (s16_r1) <br> epoch = 800, batch_size = 16 (s8_r1)   <br>    epoch = 300, batch_size = 16 (s8_r2) |      
| Optimizer                  | Momentum                                 |
| Loss Function              | Softmax Cross Entropy                                  |    
| Outputs                    | probability                                       |          
| Loss                       | 0.0065883575                                       |            
| Speed                      | 31ms/step（1pc, s8）<br> 234ms/step（8pcs, s8）                   |  
| Checkpoint for Fine tuning | 443M (.ckpt file)                                         |
| Scripts                    | [Link](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeplabv3) | 

# [Description of Random Situation](#contents)
In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py. 

# [ModelZoo Homepage](#contents)
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
 