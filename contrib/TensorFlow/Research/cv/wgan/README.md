## Wasserstein GAN

Wasserstein GAN https://arxiv.org/abs/1701.07875 Martin Arjovsky, Soumith Chintala, Léon Bottou

Tensorflow implementation of Wasserstein GAN.

Two versions:
- wgan.py: the original clipping method.
- wgan_v2.py: the gradient penalty method. (Improved Training of Wasserstein GANs).

How to run (an example):

for mnist dataset  

```
python wgan.py --data mnist --model dcgan

```

for lsun dataset  

1. download dataset from  obs://ma-iranb/data/wgan/lsun_bedrooms_64_64_300000.npy 
2. change ** wgan/lsun/\__init__.py ** line 62 to your lsun_bedrooms_64_64_300000.npy location
    
```
python wgan_.py --data lsun --model dcgan
```
