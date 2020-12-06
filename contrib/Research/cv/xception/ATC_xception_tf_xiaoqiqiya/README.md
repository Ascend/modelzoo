###   **XCEPTION** 


###   **概述** 

迁移Xception到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.79 | 0.7900  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
xception
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集
  	├─train.tfrecord
  	└─...
  ├─test_data 用于存放测试数据集
  	├─val.tfrecord
  	└─...
  ├─model 用于存放预训练模型
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

验证集数据\
obs://public-dataset/imagenet/valid_tf_299/val.tfrecord\
精度达标模型\
obs://xception-training/MA-model_arts_xception-11-27-13-23/pre_training_model\
预训练模型\
https://github.com/HiKapok/Xception_Tensorflow \
经测试发现预训练模型精度与论文中的精度有差距，但差距较小.


###   **train** 
加载预训练模型 \
python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./train_data  --output_path  ./model_save  --do_train True  --image_num  1281167 --class_num  1000  --batch_size  64  --epoch  10 --learning_rate  0.001   --save_checkpoints_steps  100 \

从头开始训练 \
python    run_xception.py  --model_path None  --data_path ./train_data  --output_path  ./model_save  --do_train True  --image_num  1281167 --class_num  1000  --batch_size  64  --epoch  10 --learning_rate  0.001   --save_checkpoints_steps  100


###  **eval** 

python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./test_data    --image_num  50000 --class_num  1000  --batch_size  100  
     
###  **参数解释**  
 
 model_path 加载模型的路径（例如 ./model/xception_model.ckpt）不加载预训练模型时设为None即可\
 data_path  tfrecord数据集的路径 （例如 ./train_data），只需要将所有的tfrecord文件放入其中 \
 output_path  经过fine_turn后的模型保存路径 （若文件夹不存在则会自动新建！！！）\
 do_train  是否训练，默认加载模型进行eval，如若需要加载预训练模型进行训练需将该值设为True\
 image_num 相应数据集包含图片数量\
 class_num 图片标签数目\
 batch_size  当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
 epoch  该值只在do_train 为True时有效，表示训练轮次\
 learning_rate 学习率\
 save_checkpoints_steps 保存模型的批次\

### 说明
由于imagenet数据较大，制作难度大，所以在制作过程中将imagenet分为24个tfrecord文件，放入同一文件夹内 \

	filepath = tf_data_path \
	tf_data_list = [] \
	file_list = os.listdir(filepath) \
	for i in file_list: \
		tf_data_list.append(os.path.join(filepath,i)) \
	return tf_data_list  \
以上代码主要功能就是将所有训练集的tfrecord文件路径以list的形式存入tf_data_list,读取文件时将此作为参数进行传递。
