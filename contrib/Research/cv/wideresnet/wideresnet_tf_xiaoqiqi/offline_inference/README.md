推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 单张推理时长| msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| wideresnet | cifar100  | `1*32*32*3` | `1*100`  | ~3ms | 81.46% | 81.45%| 

## 1、原始模型
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJp4gT7QYIh7dVwKMeUzsE7FrTNi/Ouub+ZsZRRylddk7t0uc0u8psis+QuU6pAFtSKOvhOwpSWjI7MMTX+8bJ+lM7DciKc4P1dPsKPQ25q1CabfEfv9ajkLr6qjIuyH0BeF4993ygi4JEa4sAYRttxHIyNqzYTjqGe4X17ymw06cnmO8T8u1mMWX0keMDHaNEqs7F8NvTwiBfRP3HLAwKiEnyD5dHUlcmIOsggsoP4b4lR2e9FCNvPIY7hhdtllWF5qi1SeMX4wRtir8UgIu2dHeeETgiFDDy+gPVXU7/lis0iUrbRwN0AkHwomtEnBffarYvrhKWWqz53fsKq0KabrlDfXICALLk5pnSGpxrClU=), 找到对应文件夹,下载对应的ckpt，使用改文件夹下面的convert脚本转成pb模型。(提取码为 66666) 

## 2、转om模型
[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJy+Hlp2BxkcUxmbUFdN90Q6KGIRPHN3qQt8VydIr6XorvLbl9gC/NjQ6rEysHnmRZ21ybPWCKR28M+fHTFUfmZuWtWT8rzzLkG8uWpDqwCtlMlokmMD+GsEMqmMO84QBfVqzYfmN77BarOKNr+npbi7KhcWcO6XXldYwVU2YQCqZ7mnAaRWew64C33u9IjahvJUNzXsA0SXx/Yb/3SLA9CqIc7Cm6zF+ZhpC8FtUX+j+EoJOQPiwCT9zthpn/9+/JNoWWz6aR67bRpjx9xLGIrWCUpNwRL7F3V9kfWqopv8g+kj3XuLVVlKRt3bTGtm0DiPxkrwlP1cGA8dQOoonuaQ==
)  (提取码为666666),找到对应的om的文件

atc转换命令参考：
```sh
atc --model=wideresnet.pb  --framework=3 --input_shape="inputx:1,32,32,3" --output=./wideresnet --out_nodes="output/BiasAdd:0" --soc_version=Ascend310
```


## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

### 4.1 下载预处理后的cifar100 val数据集

[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJ/iAlaumhjX2D00Szg8VClClY2W2aIejybGZS+gk/Upwt9Jqq3cV8Dtgrupt0GytGRuxvDe8TnzNtDwaWcCiYSAFM9hvblAO5zoZTahSA/BBszobCFhscwXDQVeMECjzZcTfQ9W3kI5PT/DXTbK+o+/Ng71YWSANB3MwnA4F0kJlge1csTY5e2s4GC/uwby4V+qMGMV/1H8lrvGd2ol97m0gPKdAZTnsIdyFl7FohGGTt8wV1VHuUPuiYmnY2BwC01avqPApyZU3Sia4BUco4HQU0FjPqvPGRMd00iSZ2Y/w009myMQKqepgJIjHuPs6xgff4tPoSJ7Cxq30mHzGrmY2D2a47r42jYnIj3KR8ZyI=)  (提取码为666666) 找到`cifar100_image.npy` 和 `cifar100_label.npy` 文件   

### 4.2 预处理

数据预处理在制作数据集时已经完成。wideresnet数据预处理为  
```txt
resize 224*224 
standardization : (img-mean) / std
```

### 4.3 执行推理和精度计算
执行shell脚本： ./start_inference.sh
```log
/mnt/sdc/data/cifar100/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/cifar100/data/9996.bin
[INFO] model execute success
Inference time: 3.369ms
/mnt/sdc/data/cifar100/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/cifar100/data/9997.bin
[INFO] model execute success
Inference time: 3.39ms
/mnt/sdc/data/cifar100/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/cifar100/data/9998.bin
[INFO] model execute success
Inference time: 3.379ms
/mnt/sdc/data/cifar100/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/cifar100/data/9999.bin
[INFO] model execute success
Inference time: 3.385ms
/mnt/sdc/data/cifar100/inference
[INFO] output data success
Inference average time without first time: 3.39 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
/mnt/sdc/data/cifar100/inference
[INFO] 推理结果生成结束
>>>>> 共 10000 测试样本    accuracy:0.814600

```
50000张Val数据集在NPU上推理的top1精度为：0.852， 论文参考精度值：0.854
