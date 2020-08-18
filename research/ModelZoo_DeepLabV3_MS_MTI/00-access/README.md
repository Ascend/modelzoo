# DeepLabV3 for MindSpore

## Introduction

DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two keypoints of DeepLabV3:Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple scales, and augmented ASPP makes image-level features available to capture long range information.  
paper：[Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation.](https://arxiv.org/pdf/1706.05587.pdf)  
reference code: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py

## Default configuration

- network structure

  Resnet101 as backbone, atrous convolution for dense feature extraction.

- preprocessing on training data：

  crop size: 513 * 513

  random scale: scale range 0.5 to 2.0

  random flip

  mean subtraction: means are [103.53, 116.28, 123.675]

- preprocessing on validation data：

  The image's long side is resized to 513, then the image is padded to 513 * 513

- training parameters：

  - Momentum: 0.9
  - LR scheduler: cosine
  - Learning rate(LR): 0.015
  - Batch size: 16
  - Weight decay: 0.0001
  - epochs: 150

## Pre-requirements

Before running code of this project，please ensure you have the following environments：

- hardware environment with Acsend AI cards.

- you can apply for Acsend AI cards by sending [this application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com.

  More MindSpore learning resources：

  - [MindSpore tutorial](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)
  - [MindSpore API](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)

## Quick start

1. Clone the respository.

   ```
   git clone xxx
   cd ModelZoo_DeepLabV3_MS_MTI/00-access
   ```

2. Install python packages in requirements.txt.

3. Download dataset and convert dataset to mindrecords.

   - Download segmentation dataset.

   - Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

     ```
     JPEGImages/00001.jpg SegmentationClassGray/00001.png
     JPEGImages/00002.jpg SegmentationClassGray/00002.png
     JPEGImages/00003.jpg SegmentationClassGray/00003.png
     JPEGImages/00004.jpg SegmentationClassGray/00004.png
     ......
     ```

   - Configure and run build_data.sh to convert dataset to mindrecords. Arguments in build_data.sh:

     ```
     --data_root                 root path of training data
     --data_lst                  list of training data(prepared above)
     --dst_path                  where mindrecords are saved
     --num_shards                number of shards of the mindrecords
     --shuffle                   shuffle or not
     ```

4. Generate config json file for 8-cards training.

   ```
   # From the root of this projectcd tools
   python get_multicards_json.py 10.111.*.*
   # 10.111.*.* is the computer's ip address.
   ```

5. Train.

   ```
   # From the root of this project
   # for single card training:
   chmod 777 train.sh
   ./train.sh
   # for 8 cards training:
   chmod 777 train_multicards.sh
   ./train_multicards.sh
   ```

   For finetuning, specify the pre-trained checkpoints by the argument --ckpt_pre_trained.

6. Evaluate.

   - Specify the checkpoints by the argument --ckpt_path in eval.sh.

   - Run eval.sh.

     Mean IoU will be printed on the screen.

7. Inference.

   - Prepare image list for Inference. See example/image.txt as an example.

   - Specify the destination path of result masks by the argument --dst_dir in infer.sh.

   - Run infer.sh

     
## Performance

DeepLabV3 was trained on Pascal VOC and SBD.  Different from original model in the paper, we first train on vocaug data,  then finetune it on voc train data with 1464 images.

### accuracy

| Network    | OS=16 | OS=8 | MS   | Flip  | mIOU  | mIOU in paper |
| ---------- | ----- | ---- | ---- | ----- | ----- | ------------- |
| deeplab_v3 | √     |      |      |       | 77.54 | 77.21    |
| deeplab_v3 |       | √    |      |       | 78.71 | 78.51    |
| deeplab_v3 |       | √    | √    |       | 79.55 |79.45   |
| deeplab_v3 |       | √    | √    | √     | 79.73 | 79.77        |

### training speed

| NPUs | Throughput        |
| ---- | ----------------- |
| 1    | 22.27 images/sec  |
| 8    | 152.06 images/sec |

