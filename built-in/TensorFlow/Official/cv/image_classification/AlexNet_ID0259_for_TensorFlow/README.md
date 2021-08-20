##### AlexNet
Implementation of AlexNet for Identifying that Wether first convolutional kernel's patterns are really bisected caused by it's parallel architecture

# 1.Intro
![img](./images/img.jpg)



이 구현의 Motivation은, [Alexnet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)에서 figure3를 보면 각기 다른 GPU를 사용해서 학습시켰을때, 한 GPU에서 학습된 48개 커널은 Color-agnostic하고 다른 GPU에서 학습된 48개 커널은 Color-specific하다는 설명이 쓰여있는데, 실제로 Kernel 시각화가 저렇게 되는지 확인해보기 위함이다. 알렉스넷 특유의 갈래별 Inconnectivity 때문에 저런 현상이 발생할수도 있다고 생각은 하는데 실제로 저렇게 되는가는 좀 의문이 생겼다. 실제로 저러면(모델의 갈래별로 커널의 역할이 양분되면) 그것도 그거 나름대로 딥러닝 해석 연구하는 분들께는 연구거리일거고, 갈래수를 3갈래 이상으로 증가시키면 어떨지도 또 궁금하다. 그리고 실제로 안 저러면 질문거리가 또 생기게 될 것 같다.


GPU 한대로도 모델을 두 갈래로 Parallel하게 나눠서 두 GPU가 이어지는 부분(3번째 Convolution layer와 전체 FC layer)만 Concat해주면 완전히 동일한 구현이 될 것이다. 
구현에 앞서 Pretrained-on-ImageNet 오픈소스들을 몇 개 찾아보았지만 논문의 구현 그대로가 아니었다. 모델을 두 갈래로 쪼갰다가 갈래별로 Parallel하게 Convolution하고 중간중간 Concat하는게 GPU 리소스가 딸리던 시절 썼던 트릭이라 여기는 건지, 찾아본 3, 4개의 구현모두 그냥 처음부터 끝까지 한 갈래로 모든 채널이 연결되어 있는 식이었다. 이런 식으로 학습된 커널들이라면 당연히 논문처럼 Color-agnostic, Color-specific으로 나뉘진 않을 것이다.



# 2.Implementation

분명 나와 비슷한 고민을 한 사람이 있을 것 같아서 구글링 계속해봤지만 결국 아무것도 찾지 못했으므로, 직접 구현을 시도해보기로 했다.

## 2.1.Dataset

이미지넷은 내가 가지고 있는 자원으로는 학습시키는게 불가능하므로 적당히 Image Scale이 크면서 구하기 쉬운 'Cat vs Dog' Dataset을 사용해서 학습시키기로 했다. 데이터셋이 다르다면 분명 커널 시각화 결과도 다르겠지만 그래도 뭔가 힌트를 얻을 수 있지 않을까 싶었다. 25000장 중 24000장을 Training set으로, 1000장을 Test set으로 썼다.

## 2.2.Data Augmentation

논문에 제안된 Augmentation 방법인 5-Part crop, Flip을 Runtime중 전처리시간을 최소화하기 위해 사전에 모두 적용하여 데이터를 10배 증강했으며, Color augmentation은 매 이미지마다 각각, 매번 Random하게 수행되어야 하므로 그 결과가 Epoch수 만큼 배가 될 것이므로 Runtime중 적용하였다.
따라서 Crop, Flip augmentation은 data_preprocessor.py에, Color augmentation은 data_loader.py에 구현되어 있다.

## 2.3.Parallel Architecture
![img](./images/img2.png)



오픈소스에 흔한 AlexNet 구현체들과는 다르게 실제 논문에 나와있는 구조 그대로 모델을 Parallel하게 나눠서 3번째 Convolution layer에서 일시적으로 Concat, 이후 FC layer에 들어가기 이전 완전히 Concat되게 하였다. 모델의 깊이와 두께, 커널사이즈 등은 완전히 논문과 동일하다.



## 2.4.Hyperparameters

Learning rate가 1/10 되었다는 점을 제외하고 아래와 같이 논문과 완전 동일하다.



Learning rate = 0.001



Momentum = 0.9



Weight decay coefficient = 0.0005



LRN_depth = 5



LRN_bias = 2



LRN_alpha = 0.0001



LRN_beta = 0.75



Dropout rate = 0.5



## 2.5.Other Methods

Relu activation, Local Response Normalization, Overlapping pooling, Dropout, Weight decay등 모든 기법을 동일하게, 2.4에서 보이는 바와 같이 동일한 계수로 적용하였고, Weight와 Bias Initialization도 논문과 동일하게 Weight의 경우 평균 0 표준편차 0.01의 Gaussian Sampling으로, Bias의 경우 2, 4, 5번째 Convolution layer와 FC layer는 1로, 나머지 layer는 0으로 초기화했다. Learning rate scheduling은 10 에폭 동안 Training Accuracy 상향이 없을때, Learning rate를 1/10하도록 했는데, 논문처럼 세 번에 걸쳐 1/10해도 처음 한 번만 성능 향상이 일어났다.



# 3.Result
## 3.1.Metrics
![img](./images/acc.png)



최종 Test Accuracy = 97.00%



최종 Training Accuracy = 99.07%


![img](./images/loss.png)



최종 Cross Entropy Error = 0.044084



## 3.2.Kernel Visualization
![img](./images/first_kernel_visualization.gif)



상위 48개는 GPU1, 하위 48개는 GPU2라고 보면 된다. 의심했던 바와 같이, Kernel 역할의 양분화(Color-agnostic과 Color-specific)는 관찰할 수 없었다. 원본논문과의 차이는 Dataset과 Leaning Rate 초기값, 그리고 학습속도에 유의미한 영향을 주는 Minibatch size정도 밖에 없는데, 해당 변인들을 바꾸더라도 Kernel 역할이 논문에 설명된 것처럼 양분화 될 것 같지는 않다. 내 구현에 실수가 있지는 않았는지 좀 더 AlexNet 구현에 대해 자세히 알아보는 한편, 내가 가진 장비로 ImageNet 학습은 무리이므로 장비를 빌릴 곳을 알아 봐야겠다.





**100-epoch checkpoint file link**


[https://drive.google.com/drive/folders/1f-EFtzLIy4AJf12JY9EVWOlWoAQ3NXrM]
