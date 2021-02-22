###   **  inceptionv3_frn ** 


###   **概述** 

迁移inceptionv3到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.789 | 0.7804  |
| Top-5 accuracy | 0.945 | 0.9404  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
iception
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集 #obs://public-dataset/imagenet/orignal/train_tf/ 
  	├─train.tfrecord
  	└─...
  ├─test_data 用于存放测试数据集 #obs://public-dataset/imagenet/orignal/valid_tf/
  	├─val.tfrecord  
  	└─...
  ├─model 用于存放预训练模型 #obs://inception-training/ptr_training_model/
  	├─inception_v3.ckpt
  	└─...
  ├─save_model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─inception_model.ckpt.data-00000-of-00001
  	├─inception_model.index
  	├─inception_model.meta
  	└─...
  ├─preprocess.py.py 数据预处理
  ├─train_inception_frn.py 训练
  ├─test_inception_frn.py.py 测试
  ├─train_1p.sh 模型的启动脚本，
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 imagenet 2012
http://www.image-net.org/

预训练模型\
本模型在inception的基础上提出新的归一化方式，为方便模型训练，在模型的训练时替换归一化方式,但保留预训练模型的卷积层参数以减少训练时长，替换后模型精度为0。

模型下载链接
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz \



### 训练过程及结果
epoch=20
batch_size=64
lr=动态调整
耗费近80小时

model_path---------------加载模型的路径（例如 ./model/xception_model.ckpt）不加载预训练模型时设为None即可  
data_path----------------tfrecord数据集的路径 （例如 ./train_data），只需要将所有的tfrecord文件放入其中 \
output_path--------------经过fine_turn后的模型保存路径 （若文件夹不存在则会自动新建！！！）\
image_num----------------相应数据集包含图片数量\
class_num----------------图片标签数目\
batch_size---------------当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
epoch--------------------该值只在do_train 为True时有效，表示训练轮次\

### 说明
由于imagenet数据较大，制作难度大，所以在制作过程中将imagenet分为24个tfrecord文件，放入同一文件夹内 \

	filepath = tf_data_path 
	tf_data_list = [] 
	file_list = os.listdir(filepath) 
	for i in file_list: 
		tf_data_list.append(os.path.join(filepath,i)) 
	return tf_data_list  
以上代码主要功能就是将所有训练集的tfrecord文件路径以list的形式存入tf_data_list,读取文件时将此作为参数进行传递。

 **offline_inference
** 
[offline_inference](https://gitee.com/xiaoqiqiyaya/modelzoo/tree/master/contrib/Research/cv/inceptionv3/inceptionv3_tf_xiaoqiqiya/offline_inference)