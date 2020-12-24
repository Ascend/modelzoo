###   **wideresnet** 


###   **概述** 

迁移wideresnet到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.7900 | 0.7998  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
xception
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集 #obs://public-dataset/imagenet/train_tf_299/ 
  	├─train.tfrecord
  	└─...
  ├─test_data 用于存放测试数据集 #obs://public-dataset/imagenet/valid_tf_299/
  	├─val.tfrecord  
  	└─...
  ├─model 用于存放预训练模型 #obs://xception-training/MA-model_arts_xception-11-27-13-23/pre_training_model/
  	├─checkpoint
  	├─xception_model.ckpt.data-00000-of-00001
  	├─xception_model.index
  	├─xception_model.meta
  	└─...
  ├─save_model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─xception_model.ckpt.data-00000-of-00001
  	├─xception_model.index
  	├─xception_model.meta
  	└─...
  ├─xception_model.py xception网络架构
  ├─run_xception.py 进行train和eval的一些逻辑操作
  ├─train_1p.sh 模型的启动脚本，其中包含两种模式，一种是加载预训练模型继续训练，另一种是重新训练（model_path=None时）
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 imagenet 2012
http://www.image-net.org/

预训练模型\
https://github.com/HiKapok/Xception_Tensorflow \
注：经测试发现预训练模型精度与论文中的精度有差距，但差距较小.

模型下载链接
https://drive.google.com/file/d/1sJCRDhaNaJAnouKKulB3YO8Hu3q91KjP/view \

经过训练精度达标模型
obs://xception-training/MA-model_arts_xception-11-27-13-23/model/

### 训练过程及结果
epoch=1
batch_size=64
lr=0.01
耗费4小时

xception eval  loss=6.13379   acc=0.7998 \
![输入图片说明](https://images.gitee.com/uploads/images/2020/1208/185828_331e9fdd_8376014.png "屏幕截图.png")


###   **train** 
加载预训练模型 \

python    run_wideresnet.py  --model_path ./model/wideresnet.ckpt   --data_path ./train_data --output_path  ./model_save   --depths 28  --ks 10 --do_train True  --image_num  50000   --batch_size  32 --epoch  10 --learning_rate  0.00016

          

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
由于imagenet数据较大，制作难度大，所以在制作过程中将imagenet分为24个tfrecord文件，放入同一文件夹内 \

	filepath = tf_data_path 
	tf_data_list = [] 
	file_list = os.listdir(filepath) 
	for i in file_list: 
		tf_data_list.append(os.path.join(filepath,i)) 
	return tf_data_list  
以上代码主要功能就是将所有训练集的tfrecord文件路径以list的形式存入tf_data_list,读取文件时将此作为参数进行传递。
