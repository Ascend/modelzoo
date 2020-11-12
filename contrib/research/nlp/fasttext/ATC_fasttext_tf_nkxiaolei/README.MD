1、原始模型https://github.com/ShawnyXiao/TextClassification-Keras, 训练10个Epoch，得到h5模型，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/FastText/fast_text_frozen.om

3、将下载的om文件放在model文件夹

4、参考https://gitee.com/ascend/modelzoo/tree/develop/contrib/msame, 编译出msame推理工具

5、使用msame推理工具，参考如下命令，发起推理性能测试： 

./msame --model model/fast_text_frozen.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 0.439450 ms
Inference average time without first time: 0.436374 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1Batch，shape:1x400，不带AIPP，平均推理性能0.436ms

6、精度测试：
精度数据集从训练脚本中的Validation数据集分离得来，25000帧，shape为1*400，标签为对应的情感倾向，\
使用训练集训练10个epoch得到的模型在GPU上，推理精度为：\
val_acc: 0.8859\
在NPU上验证精度：\
1、解压数据集：tar -zxvf fasttext_input.tar.gz

2、开始推理：bash start_inference.sh 得到推理精度：
```
24989, inference label:0, gt_label:0
24990, inference label:1, gt_label:1
24991, inference label:0, gt_label:0
24992, inference label:1, gt_label:1
24993, inference label:0, gt_label:0
24994, inference label:1, gt_label:1
24995, inference label:1, gt_label:1
24996, inference label:1, gt_label:1
24997, inference label:0, gt_label:0
24998, inference label:0, gt_label:0
24999, inference label:1, gt_label:0
```
Totol pic num: 25000, Top1 accuarcy: 0.8863

与预期一致