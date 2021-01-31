# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:
+ Online demo
	+ http://east.zxytim.com/
	+ Result example: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
	+ CAVEAT: There's only one cpu core on the demo server. Simultaneous access will degrade response time.
+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the paper's author.
+ The pre-trained model provided achieves **80.83** F1-score on ICDAR 2015
	Incidental Scene Text Detection Challenge using only training images from ICDAR 2015 and 2013.
  see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_samples&task=1&m=29855&gtv=1) for the detailed results.
+ Differences from original paper
	+ Use ResNet-50 rather than PVANET
	+ Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
	+ Use linear learning rate decay rather than staged learning rate decay
+ Speed on 720p (resolution of 1280x720) images:
	+ Now
		+ Graphic card: GTX 1080 Ti
		+ Network fprop: **~50 ms**
		+ NMS (C++): **~6ms**
		+ Overall: **~16 fps**
	+ Then
		+ Graphic card: K40
		+ Network fprop: ~150 ms
		+ NMS (python): ~300ms
		+ Overall: ~2 fps

Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please cite his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Demo](#demo)
3. [Test](#train)
4. [Train](#test)
5. [Examples](#examples)

### Installation
1. Any version of tensorflow version > 1.0 should be ok.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### GPU Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python multigpu_train.py \
--gpu_list=0  \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
or excute the shell:
```
bash train_gpu.sh
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### NPU Train
If you want to train on NPU , use this command:
```
python npu_train.py \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
or excute the shell:
```
bash train_npu.sh
```

### Contrast of Total Loss on GPU and NPU
With the same hyper-parameters of:
```
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
After finished 100000steps and takes 9hours on NPU,Total loss of NPU is very close to GPU's :\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0114/232451_0023bcbd_8432352.png "屏幕截图.png")

the blue one is NPU's total loss and the red one is GPU's total loss.

### Evaluation
Requirements:
```
apt-get install zip
pip3.7 install Polygon3
```
Please edit the "lanms/Makefile" for your own python env **python3.7-config**:
```
CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3.7-config --cflags)
LDFLAGS = $(shell python3.7-config --ldflags)
```

When train was finished, you can start evaluation by run the shell script：
```
bash eval.sh
```
Details in eval.sh：
```
export output_dir=./output
export ckpt_dir=./checkpoint/
export test_data=./ocr/ch4_test_images

mkdir  ${output_dir}
rm -rf ${output_dir}/*

python3.7 eval.py \
--test_data_path=${test_data} \
--checkpoint_path=${ckpt_dir} \
--output_dir=${output_dir}

cd ${output_dir}
zip results.zip res_img_*.txt
cd ../

python3.7 evaluation/script.py -g=./evaluation/gt.zip -s=${output_dir}/results.zip
```

### Contrast of Precision and Recall of GPU and NPU:
with the same train dataset(1000 images) and test dataset(500 images) of Icdar2015([BaiduYun link](https://pan.baidu.com/s/12qlSPPZl2a8rAIqeMAMyUA) 
) and the same hyper-parameters:
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.826     | 0.771  | 0.797 |
| NPU | 0.834     | 0.767  | 0.799 |

NPU Checkpoints: ([BaiduYun link](https://pan.baidu.com/s/19qRk67W3R4x_5wDbPwmWIA) )\
GPU Checkpoints: ([BaiduYun link](https://pan.baidu.com/s/1k77-11IJUBpXC90FpIoaqA) )

### Train with Icdar2013+Icdar2015
With the same hyper-parameters of:
```
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
```
After finished 100000steps and takes 12hours on NPU,Total loss of NPU is very close to GPU's :
![输入图片说明](https://images.gitee.com/uploads/images/2021/0118/233452_f06f1fb1_8432352.png "屏幕截图.png")

the blue one is NPU's total loss and the red one is GPU's total loss.

Contrast of Precision and Recall:\
with the same train dataset(icdar2013+icdar2015: 229+1000 images) and test dataset(icdar2015: 500 images) of Icdar2015([BaiduYun link](https://pan.baidu.com/s/1DsEqwvOagZRadPWAyZKhUw) 
) and the same hyper-parameters:

|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.842     | 0.766  | 0.803 |
| NPU | 0.842     | 0.779  | 0.809 |

NPU Checkpoints: ([BaiduYun link](https://pan.baidu.com/s/1UEBTfrC-cxpmIEII7H7Dqw) )\
GPU Checkpoints: ([BaiduYun link](https://pan.baidu.com/s/16rZs6z3YqdzNZCTF-40UDA) )

### Test
run
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```

a text file will be then written to the output path.