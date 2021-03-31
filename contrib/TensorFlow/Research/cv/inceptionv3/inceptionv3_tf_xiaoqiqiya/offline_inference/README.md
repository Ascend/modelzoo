推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| inceptionv3 | imageNet val  | `25*299*299*3` | `25*1001`  | 9ms~ | 77.24% | 77.86%| 

## 1、原始模型
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJp4gT7QYIh7dVwKMeUzsE7FrTNi/Ouub+ZsZRRylddk7t0uc0u8psis+QuU6pAFtSKOvhOwpSWjI7MMTX+8bJ+lM7DciKc4P1dPsKPQ25q1CabfEfv9ajkLr6qjIuyH0BeF4993ygi4JEa4sAYRttxHIyNqzYTjqGe4X17ymw06cnmO8T8u1mMWX0keMDHaNEqs7F8NvTwiBfRP3HLAwKiEnyD5dHUlcmIOsggsoP4b4lR2e9FCNvPIY7hhdtllWF5qi1SeMX4wRtir8UgIu2dHeeETgiFDDy+gPVXU7/lis0iUrbRwN0AkHwomtEnBffarYvrhKWWqz53fsKq0KabrlDfXICALLk5pnSGpxrClU=), 找到对应文件夹,下载对应的`ckpt`，使用该文件夹下面的`convert.py`脚本转成pb模型。(提取码为 66666) 

## 2、转om模型
[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJy+Hlp2BxkcUxmbUFdN90Q6KGIRPHN3qQt8VydIr6XorvLbl9gC/NjQ6rEysHnmRZ21ybPWCKR28M+fHTFUfmZuWtWT8rzzLkG8uWpDqwCtlMlokmMD+GsEMqmMO84QBfVqzYfmN77BarOKNr+npbi7KhcWcO6XXldYwVU2YQCqZ7mnAaRWew64C33u9IjahvJUNzXsA0SXx/Yb/3SLA9CqIc7Cm6zF+ZhpC8FtUX+j+EoJOQPiwCT9zthpn/9+/JNoWWz6aR67bRpjx9xLGIrWCUpNwRL7F3V9kfWqopv8g+kj3XuLVVlKRt3bTGtm0DiPxkrwlP1cGA8dQOoonuaQ==
)  (提取码为666666),找到对应的om的文件，另外我们还提供了batchsize为32的om和pb模型   

atc转换命令参考：

```sh
atc --model=inceptionv3.pb  --framework=3 --input_shape="inputx:25,299,299,3" --output=./inceptionv3 --out_nodes="InceptionV3/Logits/Conv2d_1c_1x1/Conv2D:0" --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

### 4.1 下载预处理后的imageNet val数据集

[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOIcahj1LHUIelVWC21u/2G0IjxwXS8iWEv3Ubufbq7b80j7Jf1kcG6/JU3s0W1xVsrpndoEe8EIlCxVIQiKOF50z908Rtud8Hr58FKSwAoBbI0ojUJZLGB0pZiM9irt/LJ/iAlaumhjX2D00Szg8VClClY2W2aIejybGZS+gk/Upwt9Jqq3cV8Dtgrupt0GytGRuxvDe8TnzNtDwaWcCiYSAFM9hvblAO5zoZTahSA/BBszobCFhscwXDQVeMECjzZcTfQ9W3kI5PT/DXTbK+o+/Ng71YWSANB3MwnA4F0kJlge1csTY5e2s4GC/uwby4V+qMGMV/1H8lrvGd2ol97m0gPKdAZTnsIdyFl7FohGGTt8wV1VHuUPuiYmnY2BwC01avqPApyZU3Sia4BUco4HQU0FjPqvPGRMd00iSZ2Y/w009myMQKqepgJIjHuPs6xgff4tPoSJ7Cxq30mHzGrmY2D2a47r42jYnIj3KR8ZyI=)  (提取码为666666) 找到`val.tfrecord` 文件   

### 4.2 预处理

使用inceptionv3_data_preprocess.py脚本,设定好路径后,
执行shell脚本: `./inceptionv3_data_preprocess.sh` 生成bin文件  

```sh
python3 resnet152_data_preprocess.py /home/HwHiAiUser/zjut/datasets/val.tfrecord  /mnt/sdc/data/imageNet
```

本次采用的是数据集成，预处理包含多个不同方法

```python
def preprocess_for_eval(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=True,
                        use_grayscale=False):
    """Prepare one image for evaluation.
    If height and width are specified it would output an image with that size by
    applying resize_bilinear.
    If central_fraction is specified it would crop the central fraction of the
    input image.
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
      central_crop: Enable central cropping of images during preprocessing for
        evaluation.
      use_grayscale: Whether to convert the image from RGB to grayscale.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if use_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_crop and central_fraction:
            image = tf.image.central_crop(
                image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image



images = images/255.
images1 = preprocess_for_eval(images, 299, 299, 0.80)
images2 = preprocess_for_eval(images, 299, 299, 0.85)
images3 = preprocess_for_eval(images, 299, 299, 0.9)
images4 = preprocess_for_eval(images, 299, 299, 0.95)
images5 = preprocess_for_eval(images, 299, 299, 0.925)
```

### 4.3 执行推理和精度计算

执行过程  ：  
10000张测试集，每张图片生成5张预处理图片。由于文件过大

首先是生成数据bin文件`python3 resnet152_data_preprocess.py /home/HwHiAiUser/zjut/datasets/val.tfrecord  /mnt/sdc/data/imageNet
 0`
```log
/mnt/sdc/data/imageNet/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/998.bin
[INFO] model execute success
Inference time: 227.646ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/999.bin
[INFO] model execute success
Inference time: 227.646ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
Inference average time without first time: 227.46 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
[INFO] 推理结果生成结束
>>>>> 共 25000 测试样本 accuracy:0.772360
```

然后执行 `./start_inference.sh`脚本

```log
/mnt/sdc/data/imageNet/inference
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:/mnt/sdc/data/imageNet/data/999.bin
[INFO] model execute success
Inference time: 227.63ms
/mnt/sdc/data/imageNet/inference
[INFO] output data success
Inference average time without first time: 227.51 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
[INFO] 推理结果生成结束
>>>>> 共 25000 测试样本    accuracy:0.772520
```
