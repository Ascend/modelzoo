###   **XCEPTION** 


###   **概述** 

迁移Xception到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.8150 | 0.8145 |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
WIDERESNET
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集 #obs://public-dataset/data/train.record 
  	├─train.record
  	└─...
  ├─test_data 用于存放测试数据集 #obs://public-dataset/data/val.record
  	├─val.record  
  	└─...
  ├─model 用于存放预训练模型 #obs://wideresnet-training/wideresnet/pre_training_model/
  	├─checkpoint
  	├─wideresnet.ckpt.data-00000-of-00001
  	├─wideresnet.index
  	├─wideresnet.meta
  	└─...
  ├─save_model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─wideresnet.ckpt.data-00000-of-00001
  	├─wideresnet.index
  	├─wideresnet.meta
  	└─...
  ├─wideresnet.py wideresnet网络架构
  ├─data_utils.py 数据处理
  ├─preprocess.py 数据预处理
  ├─run_wideresnet.py 进行train和eval的一些逻辑操作
  ├─train_1p.sh 模型的启动脚本
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 cifar100
http://www.cs.toronto.edu/~kriz/cifar.html \
有100个类，每个类包含600个图像。，每类各有500个训练图像和100个测试图像。CIFAR-100中的100个类被分成20个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）


模型下载链接
https://drive.google.com/drive/folders/1ypQKsJaCl6Qw8E2mFKGBfud73MAQRhHG?usp=sharing

经过预训练精度情况 
wideresnet eval  loss=0.92785   acc=0.7695   

obs://wideresnet-training/wideresnet/pre_training_model/

### 训练过程及结果
epoch=10
batch_size=32
lr=0.00016
耗费4小时

wideresnet eval  loss=0.75691   acc=0.8159 
![输入图片说明](https://images.gitee.com/uploads/images/2020/1224/145251_2557cddd_8376014.png "屏幕截图.png")


精度达标模型
obs://wideresnet-training/wideresnet/best_model/

###   **train** 
加载预训练模型 \

python    run_wideresnet.py  --model_path ./model/wideresnet.ckpt   --data_path ./train_data --output_path  ./model_save   --depths 28  --ks 10 --do_train True  --image_num  50000   --batch_size  32 --epoch  40 --learning_rate  0.00016

          

加载预训练模型直至精度达标耗时共计4小时左右


###  **eval** 

python    run_wideresnet.py  --model_path ./model_save/wideresnet.ckpt  --data_path ./test_data    --image_num  10000   --batch_size  100  

###  **参数解释**  
 

 model_path---------------加载模型的路径（例如 ./model/xception_model.ckpt）不加载预训练模型时设为None即可  
 data_path----------------tfrecord数据集的路径 （例如 ./train_data），只需要将所有的tfrecord文件放入其中 \
 output_path--------------经过fine_turn后的模型保存路径 （若文件夹不存在则会自动新建！！！）\
 do_train-----------------是否训练，默认加载模型进行eval，如若需要加载预训练模型进行训练需将该值设为True\
 image_num----------------相应数据集包含图片数量\
 batch_size---------------当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
 epoch--------------------该值只在do_train 为True时有效，表示训练轮次\
 learning_rate------------学习率\

### 说明
	filepath = tf_data_path 
	tf_data_list = [] 
	file_list = os.listdir(filepath) 
	for i in file_list: 
		tf_data_list.append(os.path.join(filepath,i)) 
	return tf_data_list  
以上代码主要功能就是将所有数据集的tfrecord文件路径以list的形式存入tf_data_list,读取文件时将此作为参数进行传递。\


 **

###  **offline_inference** 
** 
[offline_inference](https://gitee.com/xiaoqiqiyaya/modelzoo/blob/master/contrib/Research/cv/wideresnet/wideresnet_tf_xiaoqiqi/offline_inference/README.md)