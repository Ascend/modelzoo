推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| double_unet | cvcdb val | `24*256*320*3` | `24*256*320*2`  | 577ms~ | 87.22% | 85.50%|

## 1、原始模型
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJp4gT7QYIh7dVwKMeUzsE7FrTNi/Ouub+ZsZRRylddk7t0uc0u8psis+QuU6pAFtSKOvhOwpSWjI7MMTX+8bJ+lM7DciKc4P1dPsKPQ25q1CabfEfv9ajkLr6qjIuyH0BeF4993ygi4JEa4sAYRttxHIyNqzYTjqGe4X17ymw06cnmO8T8u1mMWX0keMDHaNEqs7F8NvTwiBfRP3HLAwKiEnyD5dHUlcmIOsggsoP4b4lR2e9FCNvPIY7hhdtllWF5qi1SeMX4wRtir8UgIu2dHeeETgiFDDy+gPVXU7/lis0iUrbRwN0AkHwomtEnBffarYvrhKWWqz53fsKq0KabrlDfXICALLk5pnSGpxrClU=), 找到对应文件夹,下载对应的`ckpt`，使用该文件夹下面的`convert.py`脚本转成pb模型。(提取码为 66666) 

## 2、转om模型
[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJy+Hlp2BxkcUxmbUFdN90Q6KGIRPHN3qQt8VydIr6XorvLbl9gC/NjQ6rEysHnmRZ21ybPWCKR28M+fHTFUfmZuWtWT8rzzLkG8uWpDqwCtlMlokmMD+GsEMqmMO84QBfVqzYfmN77BarOKNr+npbi7KhcWcO6XXldYwVU2YQCqZ7mnAaRWew64C33u9IjahvJUNzXsA0SXx/Yb/3SLA9CqIc7Cm6zF+ZhpC8FtUX+j+EoJOQPiwCT9zthpn/9+/JNoWWz6aR67bRpjx9xLGIrWCUpNwRL7F3V9kfWqopv8g+kj3XuLVVlKRt3bTGtm0DiPxkrwlP1cGA8dQOoonuaQ==
)  (提取码为666666),找到对应的om的文件

atc转换命令参考：

```sh
atc --model=doubleunet.pb  --framework=3 --input_shape="inputx:24,256,320,3" --output=./double_unet_2 --out_nodes="output/ResizeBilinear:0" --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

### 4.1 下载预处理后的cvcdb_val.h5数据集

[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJ/iAlaumhjX2D00Szg8VClClY2W2aIejybGZS+gk/Upwt9Jqq3cV8Dtgrupt0GytGRuxvDe8TnzNtDwaWcCiYSAFM9hvblAO5zoZTahSA/BBszobCFhscwXDQVeMECjzZcTfQ9W3kI5PT/DXTbK+o+/Ng71YWSANB3MwnA4F0kJlge1csTY5e2s4GC/uwby4V+qMGMV/1H8lrvGd2ol97m0gPKdAZTnsIdyFl7FohGGTt8wV1VHuUPuiYmnY2BwC01avqPApyZU3Sia4BUco4HQU0FjPqvPGRMd00iSZ2Y/w009myMQKqepgJIjHuPs6xgff4tPoSJ7Cxq30mHzGrmY2D2a47r42jYnIj3KR8ZyI=)  (提取码为666666) 找到`cvcdb_val.h5` 文件   

### 4.2 预处理

使用double_unet_data_preprocess.py脚本,设定好路径后,
执行shell脚本: `./double_unet_data_preprocess.sh` 生成bin文件  

```sh
python3 double_unet_data_preprocess.py /home/HwHiAiUser/zjut/datasets/val.tfrecord  /mnt/sdc/data/imageNet
```

```python
def label2data(y):
    if np.max(y) > 2:
        y = y / 255
    return y


def dimg2data(x):
    x = x / 255
    return x
```

考虑到总数为61张图片不足一个batch  所以


### 4.3 执行推理和精度计算

然后执行'./start_inference.sh'脚本
```log
[INFO] load model /home/HwHiAiUser/zjut/pb/1202pb/double_unet/double_unet_2.om success
[INFO] create model description success
[INFO] create model output success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/0.bin
[INFO] model execute success
Inference time: 577.864ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/1.bin
[INFO] model execute success
Inference time: 577.551ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/2.bin
[INFO] model execute success
Inference time: 577.22ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
Inference average time without first time: 577.39 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
[INFO] 推理结果生成结束
13
>>>>> 共 61 测试样本    accuracy:0.872276

```
