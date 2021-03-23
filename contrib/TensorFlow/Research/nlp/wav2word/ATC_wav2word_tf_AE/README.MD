## 模型功能

将语音转换为文本。

## 原始模型

参考实现 ：

https://gitee.com/li_jian_jlu/smartvoice_classnotes_c73

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Wav2word/Wav2word.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Wav2word/Wav2word_framework_tensorflow_aipp_0_batch_1_input_fp32_output_FP32.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --model="./Wav2word.pb" --output_type=FP32  --output="./Wav2word_framework_tensorflow_aipp_0_batch_1_input_fp32_output_FP32" --framework=3  --soc_version=Ascend310
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

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

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160535_390d5e33_8113712.png "图片5.png")