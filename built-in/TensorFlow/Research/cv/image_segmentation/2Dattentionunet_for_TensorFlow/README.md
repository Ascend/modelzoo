# Road_extraction
Attention Unet and Deep Unet implementation for road extraction using multi-gpu model tensorflow

Several variations of Deep U-Net were tested with extra layers and extra convolutions. Nevertheless, the model that outperformed all of them was the Attention U-Net: Learning Where to Look for the Pancreas. I have added an extra tweak improving even further performance, switching the convolution blocks to the residual blocks

# TensorFlow Segmentation
TF segmentation models, U-Net, Attention Unet, Deep U-Net (All variations of U-Net)

Image Segmentation using neural networks (NNs), designed for extracting the road network from remote sensing imagery and it can be used in other applications labels every pixel in the image (Semantic segmentation) 

Details can be found in these papers:

* [Unet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

## Attention U-Net extra module

![AU-Net](Images/aunet.png)


## Requirements
* Python 3.6
* CUDA 10.0
* TensorFlow 1.9
* Keras 2.0


## Modules
utils.py and helper.py 
functions for preprocessing data and saving it.


## Trainig model:
```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--save SAVE] [--gpu GPU]
                [--checkpoint CHECKPOINT] [--class_balancing CLASS_BALANCING]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--batch_size BATCH_SIZE] [--one_hot_label ONE_HOT_LABEL]
                [--data_aug DATA_AUG] [--change CHANGE] [--height HEIGHT]
                [--width WIDTH] [--channels CHANNELS] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --save SAVE           Interval for saving weights
  --gpu GPU             Choose GPU device to be used
  --checkpoint CHECKPOINT
                        Checkpoint folder.
  --class_balancing CLASS_BALANCING
                        Whether to use median frequency class weights to
                        balance the classes in the loss
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --one_hot_label ONE_HOT_LABEL
                        One hot label encoding
  --data_aug DATA_AUG   Use or not augmentation
  --change CHANGE       Double image 256, 512
  --height HEIGHT       Height of input image to network
  --width WIDTH         Width of input image to network
  --channels CHANNELS   Number of channels of input image to network
  --model MODEL         The model you are using. Currently supports:
                        fusionNet, fusionNet2, unet, fusionnet_atten, temp,
                        vgg_unet, fusionnet_ppl
