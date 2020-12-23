## 1、原始模型
https://github.com/argman/EAST, 下载对应的ckpt，转成pb模型

## 2、转om模型
obs链接：obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/EAST/east_text_detection.om

ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.10.0.B023/atc/bin/atc --output_type=FP32 --input_shape="input_images:1,448,448,3" --check_report=/root/modelzoo/east_text_detection/device/network_analysis.report --input_format=NHWC --output="/root/modelzoo/east_text_detection/device/east_text_detection" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/Orignal_Model/east_text_detection.pb" 
```

## 3、将下载的om文件放在model文件夹

## 4、安装dnmetis推理工具
参考https://gitee.com/ascend/tools/tree/master/dnmetis, 安装离线推理python工具

## 5、精度测试：
暂未使用cidar2015数据集进行NPU精度测试

## 6、图片推理：
### 1）安装依赖
按照requirements.txt安装python3第三方库依赖

### 2）下载图片
下载包含英文字符的图片（支持后缀.jpg、.JPEG、.png），存放在image_input目录中\
图片预处理：\
因为448x448分辨率的识别效果不佳，故参考第2步，将输入分辨率固定为：768x768 \
因为NPU的推理不支持动态Shape，所以固定输入shape为1x768x768x3，输入图片在preprocess.py中按照这个尺寸进行等比缩放和补齐
```
    if h <= resize_h and w <= resize_w:
        im = cv2.copyMakeBorder(im,0,resize_h-h,0,resize_w-w,cv2.BORDER_CONSTANT,value=(0,0,0))
        ratio_h = 1
        ratio_w = 1
    else:
        ratio_w = ratio_h = resize_h/max(h,w)
        im = cv2.resize(im, (math.floor(w*ratio_w), math.floor(h*ratio_h)))
        im = cv2.copyMakeBorder(im, 0, resize_h - math.floor(h*ratio_h), 0, resize_w - math.floor(w*ratio_w), cv2.BORDER_CONSTANT, value=(0, 0, 0))
```

### 3）执行推理
```
bash run_inference.sh
```
输出的图片和检测框信息存放在image_output目录
```
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:input//test4.jpg.bin
[INFO] model execute success
Inference time: 37.976ms
output/
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:input//test5.jpg.bin
[INFO] model execute success
Inference time: 37.992ms
output/
[INFO] output data success
[INFO] create model output success
[INFO] start to process file:input//test6.jpg.bin

start to process image_input//test1.jpg
origin_h:301, resize_h:768
origin_w:500, resize_w:768
ratio_h=ratio_w= 1
185 text boxes before nms
start to process image_input//test2.jpg
origin_h:889, resize_h:768
origin_w:500, resize_w:768
ratio_h=ratio_w= 0.8638920134983127
1296 text boxes before nms
start to process image_input//test4.jpg
origin_h:812, resize_h:768
origin_w:640, resize_w:768
ratio_h=ratio_w= 0.9458128078817734
1964 text boxes before nms
start to process image_input//test5.jpg
origin_h:499, resize_h:768
origin_w:750, resize_w:768
ratio_h=ratio_w= 1
194 text boxes before nms
```
NPU推理时间为37ms

实测效果图：\
对纯文本图片的检测:\
![对纯文本图片的检测](https://images.gitee.com/uploads/images/2020/1109/185333_1aadf2fa_7866623.jpeg "test2.jpg")

![对纯文本图片的检测](https://images.gitee.com/uploads/images/2020/1109/203854_11174a0b_7866623.jpeg "test11.jpg")

对复杂街景图片的检测:\
![对复杂街景图片的检测](https://images.gitee.com/uploads/images/2020/1109/185408_2de62a8e_7866623.jpeg "test3.jpg")