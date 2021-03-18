# DeepLab: Deep Labelling for Semantic Image Segmentation
原始模型参考[github链接](https://github.com/tensorflow/models/tree/master/research/deeplab),迁移训练代码到NPU

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- pip install tf-slim

## 代码路径解释
```shell
├─src										源码目录
│  ├─best								经过训练后最好的结果
│  ├─cp									
│  │  └─xception_65_coco_pretrained		预训练模型
│  ├─deeplab								主源码目录
│  │  ├─core							
│  │  ├─datasets						
│  │  │  └─pascal_voc_seg				数据集存放目录
│  │  │      └─tfrecord				已存放转换后的trainval、test数据集
│  │  ├─deprecated
│  │  ├─evaluation
│  │  │  ├─g3doc
│  │  │  │  └─img
│  │  │  └─testdata
│  │  │      ├─coco_gt
│  │  │      └─coco_pred
│  │  ├─g3doc
│  │  │  └─img
│  │  ├─testing
│  │  │  └─pascal_voc_seg
│  │  └─utils
│  └─slim								
│      ├─datasets
│      ├─deployment
│      ├─nets
│      │  ├─mobilenet
│      │  │  └─g3doc
│      │  └─nasnet
│      ├─preprocessing
│      └─scripts
└─tensorboard_bs7							batchsize=7下训练200k次的日志
    ├─gpu
    └─npu
```

## 准备数据和Backbone模型
可以选择自行下载 并存放到相应目录下  
pascal voc 2012 链接: https://pan.baidu.com/s/1HENumDUlkTVlul1rWQd1LQ 提取码: 6xnf  
预训练模型使用[xception_65_imagenet_coco](http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz) 




## NPU训练
在NPU上面，启动训练，使用下面的命令:
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python deeplab/train.py    \
	--logtostderr    \
	--training_number_of_steps=200000    \
	--train_split="trainval"    \
	--model_variant="xception_65"   \
	--atrous_rates=6    \
	--atrous_rates=12    \
	--atrous_rates=18    \
	--output_stride=16    \
	--decoder_output_stride=4     \
	--train_crop_size="513,513"    \
	--train_batch_size=16    \
	--dataset="pascal_voc_seg"     \
	--train_logdir=log    \
	--dataset_dir=deeplab/datasets/pascal_voc_seg/tfrecord    \
	--tf_initial_checkpoint=cp/xception_65_coco_pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt \
	--fine_tune_batch_norm=False
```
或者直接执行shell:
```
bash train.sh
```

### TotalLoss趋势比对（NPU vs GPU）
数据集和超参相同时:
```
--training_number_of_steps=300000    
--train_split="trainval"    
--model_variant="xception_65"   
--atrous_rates=6    
--atrous_rates=12    
--atrous_rates=18    
--output_stride=16    
--decoder_output_stride=4     
--train_crop_size="513,513"  
--train_batch_size=7
```
20w个Step，NPU大概花费27小时，TotalLoss收敛趋势基本一致 :\
https://pan.baidu.com/s/1v05LbDMFGmcRr0MhoG_M6Q 提取码: nxhq  

![输入图片说明](https://gitee.com/aioe/modelzoo/raw/deeplab_/contrib/TensorFlow/Research/cv/deeplab-v3-plus/deeplabv3+_hw09124698/img/vs.png "vs.png")  

蓝色是NPU，红色是GPU.

### 精度评估

等训练20w个step结束之后，可以使用vis.py来评估模型的精度，使用voc 2012测试集：
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python deeplab/vis.py \
	--logtostderr    \ 
	--vis_split="test"    \
	--model_variant="xception_65"    \
	--atrous_rates=12    \
	--atrous_rates=24   \
	--atrous_rates=36   \
	--output_stride=8   \
	--decoder_output_stride=4   \
	--vis_crop_size="513,513"    \
	--dataset="pascal_voc_seg"   \
	--checkpoint_dir=log    \
	--vis_logdir=log_e    \
	--dataset_dir=deeplab/datasets/pascal_voc_seg/tfrecord \
	--also_save_raw_predictions=True
```
当显示Finished visualization后即为完成推理 可以ctrl+c手动结束脚本！
然后进行推理结果打包

```
cd log_e
tar -zcvf results.tgz results
```
得到results.tgz后进入 http://host.robots.ox.ac.uk:8080/ 注册, 然后进入[评估页面](http://host.robots.ox.ac.uk:8080/eval/upload/)  

填入对应信息后选择results.tgz上传  

![输入图片说明](https://gitee.com/aioe/modelzoo/raw/deeplab_/contrib/TensorFlow/Research/cv/deeplab-v3-plus/deeplabv3+_hw09124698/img/upload.png "upload.png")  

稍等片刻后刷新即可得到精度结果  

![输入图片说明](https://gitee.com/aioe/modelzoo/raw/deeplab_/contrib/TensorFlow/Research/cv/deeplab-v3-plus/deeplabv3+_hw09124698/img/best.png "best.png")  



### 精度比对:
[NPU 87.72010%](http://host.robots.ox.ac.uk:8080/anonymous/AFSMJC.html)  
[论文 87.84490%](http://host.robots.ox.ac.uk:8080/anonymous/NU9OS6.html)  
NPU Checkpoints: [提取码cwsn](https://pan.baidu.com/s/1IcXF0ThsAygWZ5yjWhCd8g)  
