# NIMA: Neural Image Assessment
Implementation of [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424) in Keras + Tensorflow with weights for MobileNet model trained on AVA dataset.

NIMA assigns a Mean + Standard Deviation score to images, and can be used as a tool to automatically inspect quality of images or as a loss function to further improve the quality of generated images.

Contains weights trained on the AVA dataset for the following models:
- NASNet Mobile (0.067 EMD on valset thanks to [@tfriedel](https://github.com/tfriedel) !, 0.0848 EMD with just pre-training)
- Inception ResNet v2 (~ 0.07 EMD on valset, thanks to [@tfriedel](https://github.com/tfriedel) !)
- MobileNet (0.0804 EMD on valset)

# Usage
## Evaluation
There are `evaluate_*.py` scripts which can be used to evaluate an image using a specific model. The weights for the specific model must be downloaded from the [Releases Tab](https://github.com/titu1994/neural-image-assessment/releases) and placed in the weights directory.

Supports either passing a directory using `-dir` or a set of full paths of specific images using `-img` (seperate multiple image paths using spaces between them)

Supports passing an argument `-resize "true/false"` to resize each image to (224x224) or not before passing for NIMA scoring. 
**Note** : NASNet models do not support this argument, all images **must be resized prior to scoring !**

### Arguments: 
```
-dir    : Pass the relative/full path of a directory containing a set of images. Only png, jpg and jpeg images will be scored.
-img    : Pass one or more relative/full paths of images to score them. Can support all image types supported by PIL.
-resize : Pass "true" or "false" as values. Resize an image prior to scoring it. Not supported on NASNet models.
```

## Training
The AVA dataset is required for training these models. I used 250,000 images to train and the last 5000 images to evaluate (this is not the same format as in the paper).

First, ensure that the dataset is clean - no currupted JPG files etc by using the `check_dataset.py` script in the utils folder. If such currupted images exist, it will drastically slow down training since the Tensorflow Dataset buffers will constantly flush and reload on each occurance of a currupted image.

Then, there are two ways of training these models.

### Direct-Training
In direct training, you have to ensure that the model can be loaded, trained, evaluated and then saved all on a single GPU. If this cannot be done (because the model is too large), refer to the Pretraining section.

Use the `train_*.py` scripts for direct training. Note, if you want to train other models, copy-paste a train script and only edit the `base_model` creation part, everythin else should likely be the same.

####训练结果（train_mobilenet.py）
#####GPU
......
Epoch 17/20
  54/1250 [>.............................] - ETA: 5:24 - loss: 0.09422021-03-13 19:36:29.156783: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 353/1250 [=======>......................] - ETA: 4:53 - loss: 0.09362021-03-13 19:38:10.205784: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 394/1250 [========>.....................] - ETA: 4:40 - loss: 0.09372021-03-13 19:38:23.867902: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 836/1250 [===================>..........] - ETA: 2:21 - loss: 0.09222021-03-13 19:41:00.204429: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 935/1250 [=====================>........] - ETA: 1:47 - loss: 0.09262021-03-13 19:41:33.180570: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1061/1250 [========================>.....] - ETA: 1:04 - loss: 0.09272021-03-13 19:42:15.218290: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1250/1250 [==============================] - 438s 350ms/step - loss: 0.0929 - val_loss: 0.0905aneous bytes before marker 0xd9
Epoch 18/20
  57/1250 [>.............................] - ETA: 5:22 - loss: 0.09332021-03-13 19:43:47.591627: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 356/1250 [=======>......................] - ETA: 4:52 - loss: 0.09342021-03-13 19:45:28.787194: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 397/1250 [========>.....................] - ETA: 4:39 - loss: 0.09332021-03-13 19:45:42.346459: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 839/1250 [===================>..........] - ETA: 2:20 - loss: 0.09182021-03-13 19:48:18.600632: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 938/1250 [=====================>........] - ETA: 1:46 - loss: 0.09222021-03-13 19:48:51.592647: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1064/1250 [========================>.....] - ETA: 1:03 - loss: 0.09262021-03-13 19:49:33.606246: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1250/1250 [==============================] - 437s 350ms/step - loss: 0.0925 - val_loss: 0.0963aneous bytes before marker 0xd9
Epoch 19/20
  60/1250 [>.............................] - ETA: 5:32 - loss: 0.09442021-03-13 19:51:06.211445: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 359/1250 [=======>......................] - ETA: 4:52 - loss: 0.09332021-03-13 19:52:47.402093: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 400/1250 [========>.....................] - ETA: 4:38 - loss: 0.09332021-03-13 19:53:00.962551: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 842/1250 [===================>..........] - ETA: 2:19 - loss: 0.09192021-03-13 19:55:37.317360: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 941/1250 [=====================>........] - ETA: 1:45 - loss: 0.09192021-03-13 19:56:10.297084: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1067/1250 [========================>.....] - ETA: 1:02 - loss: 0.09212021-03-13 19:56:52.512312: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1250/1250 [==============================] - 438s 350ms/step - loss: 0.0923 - val_loss: 0.1013aneous bytes before marker 0xd9
Epoch 20/20
  63/1250 [>.............................] - ETA: 5:31 - loss: 0.09512021-03-13 19:58:24.751516: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 362/1250 [=======>......................] - ETA: 4:50 - loss: 0.09312021-03-13 20:00:05.941336: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 403/1250 [========>.....................] - ETA: 4:37 - loss: 0.09332021-03-13 20:00:19.566188: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 845/1250 [===================>..........] - ETA: 2:18 - loss: 0.09132021-03-13 20:02:55.762785: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
 944/1250 [=====================>........] - ETA: 1:44 - loss: 0.09202021-03-13 20:03:28.723488: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1070/1250 [========================>.....] - ETA: 1:01 - loss: 0.09252021-03-13 20:04:10.887920: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
1250/1250 [==============================] - 437s 350ms/step - loss: 0.0928 - val_loss: 0.0871aneous bytes before marker 0xd9

#####NPU
1）将25w数据集images放到AVA_dataset
2）将mobilenet_1_0_224_tf_no_top.h5放到环境上/root/.keras/models/目录
3）执行./run_npu_1p.sh拉起训练

......
Epoch 17/20
  38/1250 [..............................] - ETA: 10:57 - loss: 0.08952021-03-13 21:44:04.409672: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  337/1250 [=======>......................] - ETA: 10:44 - loss: 0.08792021-03-13 21:47:42.058850: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  379/1250 [========>.....................] - ETA: 10:15 - loss: 0.08802021-03-13 21:48:11.464911: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  820/1250 [==================>...........] - ETA: 5:15 - loss: 0.08792021-03-13 21:53:44.974518: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  919/1250 [=====================>........] - ETA: 4:01 - loss: 0.08802021-03-13 21:54:55.953082: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1045/1250 [========================>.....] - ETA: 2:29 - loss: 0.08792021-03-13 21:56:26.122680: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1250/1250 [==============================] - 941s 753ms/step - loss: 0.0880 - val_loss: 0.0892ot improvetes before marker 0xd9
Epoch 18/20
  40/1250 [..............................] - ETA: 11:07 - loss: 0.08922021-03-13 21:59:46.851367: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  339/1250 [=======>......................] - ETA: 10:41 - loss: 0.08802021-03-13 22:03:24.049062: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  381/1250 [========>.....................] - ETA: 10:13 - loss: 0.08812021-03-13 22:03:53.741240: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  822/1250 [==================>...........] - ETA: 5:13 - loss: 0.08792021-03-13 22:09:27.669435: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  921/1250 [=====================>........] - ETA: 4:00 - loss: 0.08812021-03-13 22:10:38.631871: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1047/1250 [========================>.....] - ETA: 2:28 - loss: 0.08792021-03-13 22:12:08.649918: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1249/1250 [============================>.] - ETA: 0s - loss: 0.0880Epoch 00018: val_loss improved from 0.08919 to 0.08918, saving model to weight
  1250/1250 [==============================] - 941s 753ms/step - loss: 0.0880 - val_loss: 0.0892
Epoch 19/20
  42/1250 [>.............................] - ETA: 11:11 - loss: 0.08922021-03-13 22:15:29.202636: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  341/1250 [=======>......................] - ETA: 10:41 - loss: 0.08802021-03-13 22:19:07.068710: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  383/1250 [========>.....................] - ETA: 10:12 - loss: 0.08812021-03-13 22:19:36.273640: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  824/1250 [==================>...........] - ETA: 5:12 - loss: 0.08792021-03-13 22:25:10.319390: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  923/1250 [=====================>........] - ETA: 3:59 - loss: 0.08802021-03-13 22:26:21.535016: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1049/1250 [========================>.....] - ETA: 2:26 - loss: 0.08792021-03-13 22:27:51.635784: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1249/1250 [============================>.] - ETA: 0s - loss: 0.0879Epoch 00019: val_loss improved from 0.08918 to 0.08918, saving model to weight
  1250/1250 [==============================] - 942s 753ms/step - loss: 0.0879 - val_loss: 0.0892
Epoch 20/20
  44/1250 [>.............................] - ETA: 11:26 - loss: 0.09032021-03-13 22:31:12.441754: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  343/1250 [=======>......................] - ETA: 10:40 - loss: 0.08812021-03-13 22:34:50.173588: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  385/1250 [========>.....................] - ETA: 10:11 - loss: 0.08812021-03-13 22:35:19.622496: W tensorflow/core/lib/png/png_io.cc:88] PNG war 
  826/1250 [==================>...........] - ETA: 5:11 - loss: 0.08802021-03-13 22:40:53.458155: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  925/1250 [=====================>........] - ETA: 3:57 - loss: 0.08802021-03-13 22:42:04.684432: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1051/1250 [========================>.....] - ETA: 2:25 - loss: 0.08802021-03-13 22:43:35.306572: W tensorflow/core/lib/png/png_io.cc:88] PNG warn
  1249/1250 [============================>.] - ETA: 0s - loss: 0.0879Epoch 00020: val_loss improved from 0.08918 to 0.08918, saving model to weight
  1250/1250 [==============================] - 942s 754ms/step - loss: 0.0879 - val_loss: 0.0892

### Pre-Training
If the model is too large to train directly, training can still be done in a roundabout way (as long as you are able to do inference with a batch of images with the model).

**Note** : One obvious drawback of such a method is that it wont have the performance of the full model without further finetuning. 

This is a 3 step process:

1)  **Extract features from the model**: Use the `extract_*_features.py` script to extract the features from the large model. In this step, you can change the batch_size to be small enough to not overload your GPU memory, and save all the features to 2 TFRecord objects.

2) **Pre-Train the model**: Once the features have been extracted, you can simply train a small feed forward network on those features directly. Since the feed forward network will likely easily fit onto memory, you can use large batch sizes to quickly train the network.

3) **Fine-Tune the model**: This step is optional, only for those who have sufficient memory to load both the large model and the feed forward classifier at the same time. Use the `train_nasnet_mobile.py` as reference as to how to load both the large model and the weights of the feed forward network into this large model and then train fully for several epochs at a lower learning rate.

# Example
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>

<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA2.jpg?raw=true" height=100% width=100%>




# Requirements
- Keras==2.1.2
- Tensorflow (1.15)
- Numpy
- Path.py
- PIL
-h5py<3.0.0
