## 1、原始模型
用户如果想自行转换om文件，可下载原生Tensorflow模型（.pb）及AIPP配置文件，若不想自行转换，可直接跳到 [2转om模型](#转om模型) 

obs链接: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Wav2word/Wav2word.pb

下载后，使用ATC模型转换工具或者执行转换命令进行转换：
```
atc --model="./Wav2word.pb" --output_type=FP32  --output="./Wav2word_framework_tensorflow_aipp_0_batch_1_input_fp32_output_FP32" --framework=3  --soc_version=Ascend310
```


## 2、转om模型
若用户不想手动进行转换，可下载转换好的om模型

obs链接：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Wav2word/Wav2word_framework_tensorflow_aipp_0_batch_1_input_fp32_output_FP32.om



## 3、将下载的om文件放在model文件夹

## 4、编译msame推理工具
参考 https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

## 5、性能测试
使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/Wav2word_framework_tensorflow_aipp_0_batch_1_input_fp32_output_FP32.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```
```
[INFO] output data success
Inference average time: 7.613180 ms
Inference average time without first time: 7.611788 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batch: 1, shape: 1600 * 200 * 1，不带AIPP，平均推理性能7.61ms

## 6、精度测试
待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160535_390d5e33_8113712.png "图片5.png")