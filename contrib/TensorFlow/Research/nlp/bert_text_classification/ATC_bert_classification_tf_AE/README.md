## 模型功能

通过对输入的新闻文本进行分类。

## 原始模型

参考实现 ：

https://github.com/percent4/keras_bert_text_classification

原始模型网络及权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/bert_text_classification/bert_text_classification.pb


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/bert_text_classification/bert_text_classification.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=bert_text_classification.pb --framework=3 --input_format="ND" --output=bert_text_classification --input_shape="input_1:1,300;input_2:1,300" --out_nodes=dense_1/Softmax:0 --soc_version=Ascend310 --op_select_implmode="high_precision"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/bert_text_classification.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 80.482810 ms
Inference average time without first time: 80.482040 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 300，不带AIPP，平均推理性能80.482040ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0303/153732_532fea37_5302634.png "屏幕截图.png")
