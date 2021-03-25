推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度MIoU | 目标精度MIoU |
|--|--|--|---| -- | --| -- |
| fcn | voc2011  | `1*512*512*3` | `1*520*520*21`  | 247ms~ | 63.34% | 63.34%| 

## 1、原始模型
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJp4gT7QYIh7dVwKMeUzsE7FrTNi/Ouub+ZsZRRylddk7t0uc0u8psis+QuU6pAFtSKOvhOwpSWjI7MMTX+8bJ+lM7DciKc4P1dPsKPQ25q1CabfEfv9ajkLr6qjIuyH0BeF4993ygi4JEa4sAYRttxHIyNqzYTjqGe4X17ymw06cnmO8T8u1mMWX0keMDHaNEqs7F8NvTwiBfRP3HLAwKiEnyD5dHUlcmIOsggsoP4b4lR2e9FCNvPIY7hhdtllWF5qi1SeMX4wRtir8UgIu2dHeeETgiFDDy+gPVXU7/lis0iUrbRwN0AkHwomtEnBffarYvrhKWWqz53fsKq0KabrlDfXICALLk5pnSGpxrClU=), 找到对应文件夹,下载对应的`ckpt`，使用该文件夹下面的`convert.py`脚本转成pb模型。(提取码为 66666) 

## 2、转om模型
[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJy+Hlp2BxkcUxmbUFdN90Q6KGIRPHN3qQt8VydIr6XorvLbl9gC/NjQ6rEysHnmRZ21ybPWCKR28M+fHTFUfmZuWtWT8rzzLkG8uWpDqwCtlMlokmMD+GsEMqmMO84QBfVqzYfmN77BarOKNr+npbi7KhcWcO6XXldYwVU2YQCqZ7mnAaRWew64C33u9IjahvJUNzXsA0SXx/Yb/3SLA9CqIc7Cm6zF+ZhpC8FtUX+j+EoJOQPiwCT9zthpn/9+/JNoWWz6aR67bRpjx9xLGIrWCUpNwRL7F3V9kfWqopv8g+kj3XuLVVlKRt3bTGtm0DiPxkrwlP1cGA8dQOoonuaQ==
)  (提取码为666666),找到对应的om的文件，另外我们还提供了batchsize为32的om和pb模型   

atc转换命令参考：

```sh
atc --model=fcn.pb  --framework=3 --input_shape="inputx:1,512,512,3" --output=./fcn --out_nodes="upsample/conv2d_transpose_2/conv2d_transpose:0" --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

### 4.1 下载预处理后的imageNet val数据集

[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJ/iAlaumhjX2D00Szg8VClClY2W2aIejybGZS+gk/Upwt9Jqq3cV8Dtgrupt0GytGRuxvDe8TnzNtDwaWcCiYSAFM9hvblAO5zoZTahSA/BBszobCFhscwXDQVeMECjzZcTfQ9W3kI5PT/DXTbK+o+/Ng71YWSANB3MwnA4F0kJlge1csTY5e2s4GC/uwby4V+qMGMV/1H8lrvGd2ol97m0gPKdAZTnsIdyFl7FohGGTt8wV1VHuUPuiYmnY2BwC01avqPApyZU3Sia4BUco4HQU0FjPqvPGRMd00iSZ2Y/w009myMQKqepgJIjHuPs6xgff4tPoSJ7Cxq30mHzGrmY2D2a47r42jYnIj3KR8ZyI=)  (提取码为666666) 找到`fcn.tfrecord` 文件   

### 4.2 预处理

使用fcn_data_preprocess.py脚本,设定好路径后,
执行shell脚本: `./fcn_data_preprocess.sh` 生成bin文件  

```sh
python3 fcn_data_preprocess.py /home/HwHiAiUser/zjut/datasets/fcn.tfrecord  /mnt/sdc/data/fcn
```

```python
image_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0))
label_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2))
image = tf.pad(image, image_padding, mode='constant', constant_values=0)
label = tf.pad(label, label_padding, mode='constant', constant_values=0)
image = image - [122.67891434, 116.66876762, 104.00698793]
image = image / 255.
```

### 4.3 执行推理和精度计算

执行'./start_inference.sh'脚本
```log
/mnt/sdc/data/fcn/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/fcn/data/97.bin
[INFO] model execute success
Inference time: 246.922ms
/mnt/sdc/data/fcn/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/fcn/data/98.bin
[INFO] model execute success
Inference time: 246.886ms
/mnt/sdc/data/fcn/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/fcn/data/99.bin
[INFO] model execute success
Inference time: 247.124ms
/mnt/sdc/data/fcn/inference
[INFO] output data success
Inference average time without first time: 246.93 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
/mnt/sdc/data/fcn/inference
[INFO] 推理结果生成结束
>>>>> 共 736 测试样本    MIoU: 0.633414
```
