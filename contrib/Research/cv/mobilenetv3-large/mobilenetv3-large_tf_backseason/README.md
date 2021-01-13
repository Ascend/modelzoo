# MobileNetv3 for Tensorflow 

This repository is based on [MobileNetV2](https://www.huaweicloud.com/ascend/resources/modelzoo/Model%20Scripts/9e7e42efd75f4d8eb4ee501639fb8090) and provides recipes to train the MobileNetV3_Large model on four GPUs and then finetune it on one NPU, which achieves a Top-1 Acc of ~74.1.

## Model overview

In this repository, we implement MobileNetV3_Large from paper 
[Howard, Andrew et al. "Searching for mobilenetv3." ICCV 2019.](https://arxiv.org/abs/1905.02244). More details can be found in the paper.

### Training configuration

We introduce the default configurations and hyperparameters we used to train the MobileNetV3_Large model.

#### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum: 0.9
- Learning rate (LR): 0.3
- LR schedule: cosine_annealing
- Warmup epoch: 4
- Batch size: 320\*4
- Weight decay:  0.00004 
- Moving average decay: 0.9999
- Learning rate decay factor: 0.99
- Number of epochs per decay: 1.0
- Label smoothing: 0.1
- Max epoch: 1200

#### Data augmentation

This model uses the data augmentation from InceptionV2:

- For training:
  - Convert DataType and RandomResizeCrop
  - RandomHorizontalFlip, prob=0.5
  - Subtract with 0.5 and multiply with 2.0
- For inference:
  - Convert DataType
  - CenterCrop 87.5% of the original image and resize to (224, 224)
  - Subtract with 0.5 and multiply with 2.0

For more details, we refer readers to read the corresponding source code in [slim](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

## Setup
The following section introduces the detailed steps to train the model on GPUs and then use an intermediate checkpoint to finetune it on NPUs further.

### Requirements

Tensorflow 1.15.0

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/modelzoo/tree/master/contrib/Research/cv/mobilenetv3-large/mobilenetv3-large_tf_backseason
cd gpu
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset
2. Generate tfrecord files following [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).
3. The train and validation tfrecord files are under the path/data directories.

### 3. Set environment

Set environment variable like *LD_LIBRARY_PATH*, *PYTHONPATH* and *PATH* to match your system before training and testing.

### 4. Train on GPUs
We tested on 4 RTX Titan GPUs.
```shell
bash train_mobilenetv3_4p_xla_amp_yz_74_10.sh
```
After training (~1,200,000 steps, ~179hrs, Top-1 Acc=74.1) we will get the checkpoints under the `gpu/results` folder. 
[Download Link](https://backseason-mbv3-open.obs.cn-north-4.myhuaweicloud.com/mobilenetv3_large_models/gpu/results_mbv3_74_10.tar)

### 5. Test
- We evaluate results by using following commands:
     ```shell 
     python3 eval_image_classifier_mobilenet.py --dataset_dir data/ilsvrc2012_tfrecord --batch_size 256 --model_name mobilenet_v3_large --checkpoint_path results/model.ckpt-1200000
    ```
    Remember to modify the dataset path and checkpoint path, then run the command.

### 6. Finetune on NPU-Ascend 910
We need to copy the intermediate checkpoints trained on GPUs (~1,080,000 steps, Top-1 Acc=72.7) to `npu/snapshots` and modify the `npu/snapshots/checkpoint` file accordingly. [Download Link](https://backseason-mbv3-open.obs.cn-north-4.myhuaweicloud.com/mobilenetv3_large_models/npu/snapshots.tar)

Then under the `npu/` folder we begin the finetuning process on NPU using ModelArts with PyCharm toolkit.

For usage of ModelArts please refer to this [wiki](https://gitee.com/backseason/modelzoo/wikis/ModelArts+PyCharm%E4%B8%8A%E4%BD%BF%E7%94%A8NPU%E7%8E%AF%E5%A2%83%E4%B8%80%E9%94%AE%E8%84%9A%E6%9C%AC%E7%A4%BA%E4%BE%8B?sort_id=3335465).

The default configurations and hyperparameters are updated accordingly as we have only one NPU at present.

- Learning rate (LR) : 0.3 -> 0.15
- Iterations per loop: 50 
- Batch size : 320\*4 -> 640\*1
- Number of epochs per decay: 1.0 -> 2.0
- Max epoch: 1200 -> 600 (to resume from the right steps)

After fintuning on NPU (120,000 steps, ~29hrs), we can achieve a Top-1 Acc of 74.12, which is comparable to the performance of GPUs.
[Log link](https://backseason-mbv3-open.obs.cn-north-4.myhuaweicloud.com/mobilenetv3_large_models/mobilenet_v3_test_V0021_job-mobilenet-v3-test.0.log)

### 6. Inference on NPU-Ascend 310
We achieve an inference time of ~2.8ms on the Ascend 310 NPU with an overall Top-1 Acc of 74.28. 
[Download Link(pb and om models)](https://backseason-mbv3-open.obs.cn-north-4.myhuaweicloud.com/mobilenetv3_large_models/npu/ATC/pb_om_model.tar)
[Download Link(ATC logs)](https://backseason-mbv3-open.obs.cn-north-4.myhuaweicloud.com/mobilenetv3_large_models/npu/ATC/ATC_logs.tar)

For usage of offline inference on HUAWEI ECS please refer to this [wiki](https://gitee.com/backseason/modelzoo/wikis/MobileNetV3_Large_MindStudio%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B?sort_id=3335182).

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

|  Platform    |   epochs        |      Top1      |
| :----------: | :-------------: | :------------: |
| 4xRTX Titan  | 1,200 (scratch) |    74.1%       |
| 1xAscend 910 |  60 (fintune)   | 72.7% -> 74.1% |

#### Training performance results
| Platform         | batch size |  speed    |
| :--------------: | :--------: |:--------: |
| 1xAscend 910     |    192     | 1882 img/s|
| 1xTesla V100-16G |    192     | 774 img/s |
| 1xAscend 910     |    128     | 1805 img/s|
| 1xTesla V100-16G |    128     | 691 img/s |
| 1xAscend 910     |    64      | 1498 img/s|
| 1xTesla V100-16G |    64      | 525 img/s |
| 1xAscend 910     |    32      | 1110 img/s|
| 1xTesla V100-16G |    32      | 355 img/s |
| 1xAscend 910     |    1       | 68  img/s |
| 1xTesla V100-16G |    1       | 21 img/s |


