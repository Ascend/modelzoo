###   **XCEPTION** 


###   **概述** 

迁移Xception到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.79 | 0.7856  |

 ** **Requirements** ** 
1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
xception
└─ 
  ├─README.md
  ├─data 用于存放数据集
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
  ├─run.sh 模型的启动脚本，其中包含两种模式，一种是加载预训练模型继续训练，另一种是加载模型进行eval
```
###   **数据集和模型** 

验证集数据\
obs://public-dataset/imagenet/valid_tf_299/val.tfrecord\
精度达标模型\
obs://xception-training/MA-model_arts_xception-11-27-13-23/pre_training_model/

###   **train** 

python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./data/val.tfrecord  --output_path  ./model_save  --do_train True  --image_num  50000 --class_num  1000  --batch_size  64  --epoch  10 --learning_rate  0.001   --save_checkpoints_steps  100

###  **eval** 

python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./data/val.tfrecord    --image_num  50000 --class_num  1000  --batch_size  100  
     
###  **参数解释**  
 
 model_path 加载模型的路径（例如 ./model/xception_model.ckpt）\
 data_path  tfrecord数据集的路径 （例如 ./data/val.tfrecord）\
 output_path  经过fine_turn后的模型保存路径 \
 do_train  是否训练，默认加载模型进行eval，如若需要加载预训练模型进行训练需将该值设为True\
 image_num 相应数据集包含图片数量\
 class_num 图片标签数目\
 batch_size  当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
 epoch  该值只在do_train 为True时有效，表示训练轮次\
 learning_rate 学习率\
 save_checkpoints_steps 保存模型的批次\