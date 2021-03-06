# Keyword spotting for Microcontrollers 

This repository consists of the tensorflow models and training scripts used 
in the paper: 
[Hello Edge: Keyword spotting on Microcontrollers](https://arxiv.org/pdf/1711.07128.pdf). 
The scripts are adapted from [Tensorflow examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) 
and some are repeated here for the sake of making these scripts self-contained.

To train a DNN with 3 fully-connected layers with 128 neurons in each layer, run:

```
python train.py --model_architecture dnn --model_size_info 128 128 128 
```
The command line argument *--model_size_info* is used to pass the neural network layer
dimensions such as number of layers, convolution filter size/stride as a list to models.py, 
which builds the tensorflow graph based on the provided model architecture 
and layer dimensions. 
For more info on *model_size_info* for each network architecture see 
[models.py](models.py).
The training commands with all the hyperparameters to reproduce the models shown in the 
[paper](https://arxiv.org/pdf/1711.07128.pdf) are given [here](train_commands.txt).

To run inference on the trained model from a checkpoint on train/val/test set, run:
```
python test.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint 
<checkpoint path>
```

To freeze the trained model checkpoint into a .pb file, run:
```
python freeze.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint 
<checkpoint path> --output_file dnn.pb
```

## Pretrained models

Trained models (.pb files) for different neural network architectures such as DNN,
CNN, Basic LSTM, LSTM, GRU, CRNN and DS-CNN shown in 
this [arXiv paper](https://arxiv.org/pdf/1711.07128.pdf) are added in 
[Pretrained_models](Pretrained_models). Accuracy of the models on validation set, 
their memory requirements and operations per inference are also summarized in the 
following table.

<img src="https://user-images.githubusercontent.com/34459978/34018008-0451ef9a-e0dd-11e7-9661-59e4fb4a8347.png">

To run an audio file through the trained model (e.g. a DNN) and get top prediction, 
run:
```
python label_wav.py --wav <audio file> --graph Pretrained_models/DNN/DNN_S.pb 
--labels Pretrained_models/labels.txt --how_many_labels 1
```

## Quantization Guide and Deployment on Microcontrollers

A quick guide on quantizing the KWS neural network models is [here](Deployment/Quant_guide.md). 
The example code for running a DNN model on a Cortex-M development board is also provided [here](Deployment). 

## Train on NPU
Because of the issue : [Issue](https://gitee.com/ascend/modelzoo/issues/I2AMF2?from=project-issue)

Modify the input_data.py:
```
 background_add = tf.add(background_mul, sliced_foreground)
 background_clamp = background_add
 #background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
```
Train on NPU:
```
python3.7 train.py --model_architecture dnn --model_size_info 128 128 128
```
Or you can simply use the shell script:
```
bash train_npu.sh
```
Log of NPU training:
```
2020-12-24 23:11:47.268417: I tf_adapter/kernels/geop_npu.cc:573] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp23_0[ 2741664us]
INFO:tensorflow:Step #1: rate 0.001000, accuracy 3.00%, cross entropy 4.509356
I1224 23:11:47.320920 281473027969040 train.py:250] Step #1: rate 0.001000, accuracy 3.00%, cross entropy 4.509356
2020-12-24 23:11:47.326019: I tf_adapter/kernels/geop_npu.cc:388] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp21_0, num_inputs:7, num_outputs:1
2020-12-24 23:11:47.326269: I tf_adapter/kernels/geop_npu.cc:260] [GEOP] tf session direct208f13df09267953, graph id: 11 no need to rebuild
2020-12-24 23:11:47.326757: I tf_adapter/kernels/geop_npu.cc:580] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp21_0 ,tf session: direct208f13df09267953 ,graph id: 11
[TRACE] GE(10333,python3.7):2020-12-24-23:11:47.326.814 [status:RUNNING] [framework/domi/client/ge_api.cc:385]11044 RunGraphAsync:Run Graph Asynchronously
2020-12-24 23:11:47.327037: I tf_adapter/kernels/geop_npu.cc:593] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp21_0, ret_status:success ,tf session: direct208f13df09267953 ,graph id: 11 [0 ms]

```

Contrast of accuracy of CPU and NPU(200 steps):\
![输入图片说明](https://images.gitee.com/uploads/images/2020/1225/201833_4c137a8f_8432352.png "屏幕截图.png")