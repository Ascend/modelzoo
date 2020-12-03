## 1、转换模型
NoisyStudent的pb模型，下载地址来自于Google的modelzoo

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="images:1,800,800,3" --check_report=/root/modelzoo/noisy_student_efficientnet-l2/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/noisy_student_efficientnet-l2/device/noisy_student_efficientnet-l2" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/noisy_student_efficientnet-l2.pb" 
```
## 2、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/ccl/msame, 编译出msame推理工具

## 3、使用msame工具进行推理性能测试
转om模型，obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/NoisyStudent/noisy_student_efficientnet-l2.om\
将下载的om文件放在model文件夹\
使用msame推理工具，参考如下命令，发起推理性能测试： \
./msame --model model/noisy_student_efficientnet-l2.om --output output/ --loop 100
```
[INFO] output data success
Inference average time: 673.064750 ms
Inference average time without first time: 673.051556 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
1batch的平均推理时间 673.05ms


## 4、精度测试：

### 4.1 下载ImageNet 2012 val数据集

### 4.2 图片预处理
下载好的数据集JPEG图片放在任意其他目录(以originnal_pic为例)，执行预处理脚本生成bin文件：
```
python3.7.5 img_preprocess.py ./originnal_pic/
```
将生成的bin文件移动至input目录，标签文件val_map.txt放至ground_truth目录

### 4.3 执行推理和精度计算的shell脚本： 
./start_inference.sh
```
ILSVRC2012_val_00049998.JPEG, inference label:232, gt_label:232
ILSVRC2012_val_00049999.JPEG, inference label:982, gt_label:982
ILSVRC2012_val_00050000.JPEG, inference label:355, gt_label:355
```
50000张Val数据集在NPU上推理的top1精度为：0.879， 论文参考精度值：0.884