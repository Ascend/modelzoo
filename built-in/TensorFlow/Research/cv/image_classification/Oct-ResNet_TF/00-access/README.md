# octconv_resnet
origin paper https://arxiv.org/abs/1904.05049

# environment
tensorflow1.13.1  
Scipy
## recommend
docker:  
tensorflow/tensorflow:latest-gpu-py3

more detail in https://tensorflow.google.cn/install

# train
> python train.py 1 3  
> #python train.py VERSION RES_BLOCKS  
default epochs is 200 and batch_size is 32

# score
> python score.py MODELFILE
