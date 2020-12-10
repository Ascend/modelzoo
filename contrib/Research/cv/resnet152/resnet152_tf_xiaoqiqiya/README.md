###   **resnet-152** 


###   **概述** 

迁移resnet-152到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.7700 | 0.75.9  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
xception
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集 #obs://public-dataset/imagenet/train_tf_224/ 
  	├─train.tfrecord
  	└─...
  ├─test_data 用于存放测试数据集 #obs://public-dataset/imagenet/valid_tf_224/
  	├─val.tfrecord  
  	└─...
  ├─model 用于存放预训练模型 #obs://resnet-training/MA-model_art_resnet-11-26-20-34/pretrain_model/
  	├─resnet_v1_152.ckpt
  	└─...
  ├─save_model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─resnet152_model.ckpt.data-00000-of-00001
  	├─resnet152_model.index
  	├─resnet152_model.meta
  	└─...
  ├─data_utils.py 获取数据
  ├─resnet52.py resnet152网络架构
  ├─resnet_utils.py resnet网络架构
  ├─run_resnet.py 进行train和eval的一些逻辑操作
  ├─train_1p.sh 模型的启动脚本，其中包含两种模式，一种是加载预训练模型继续训练，另一种是重新训练（model_path=None时）
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 imagenet 2012
http://www.image-net.org/

预训练模型\
https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py \
注：经测试发现预训练模型精度与论文中的精度有差距，但差距较大，可能是由于数据预处理未对齐.

模型下载链接
http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz \



### 训练过程及结果
epoch=4
batch_size=128
lr=0.0001
耗费7小时


###   **train** 
加载预训练模型 \
python    run_resnet.py  --model_path ./model/resnet_v1_152.ckpt  --data_path  ./train_data --output_path  ./model_save  --do_train True  --image_num  1281167 --class_num  1000  --batch_size  128 --epoch  5  --learning_rate  0.0001   --save_checkpoints_steps  100 \
    

加载预训练模型直至精度达标耗时共计25小时左右


从头开始训练 \
python    run_resnet.py  --model_path None  --data_path  ./train_data --output_path  ./model_save  --do_train True  --image_num  1281167 --class_num  1000  --batch_size  128 --epoch  5  --learning_rate  0.0001   --save_checkpoints_steps  100
    

注：只提供该训练方式，但并未采用该方式进行训练！！

###  **eval** 

python    run_resnet.py  --model_path ./model/resnet.ckpt --data_path ./test_data/    --image_num  50000 --class_num  1000  --batch_size  100

###  **参数解释**  
 

 model_path---------------加载模型的路径（例如 ./model/xception_model.ckpt）不加载预训练模型时设为None即可  
 data_path----------------tfrecord数据集的路径 （例如 ./train_data），只需要将所有的tfrecord文件放入其中 \
 output_path--------------经过fine_turn后的模型保存路径 （若文件夹不存在则会自动新建！！！）\
 do_train-----------------是否训练，默认加载模型进行eval，如若需要加载预训练模型进行训练需将该值设为True\
 image_num----------------相应数据集包含图片数量\
 class_num----------------图片标签数目\
 batch_size---------------当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
 epoch--------------------该值只在do_train 为True时有效，表示训练轮次\
 learning_rate------------学习率\
 save_checkpoints_steps---保存模型的批次

### 说明
由于imagenet数据较大，制作难度大，所以在制作过程中将imagenet分为24个tfrecord文件，放入同一文件夹内 \

	filepath = tf_data_path 
	tf_data_list = [] 
	file_list = os.listdir(filepath) 
	for i in file_list: 
		tf_data_list.append(os.path.join(filepath,i)) 
	return tf_data_list  
以上代码主要功能就是将所有训练集的tfrecord文件路径以list的形式存入tf_data_list,读取文件时将此作为参数进行传递。