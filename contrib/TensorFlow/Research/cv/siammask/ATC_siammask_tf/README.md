# SiamMask

#### 1、模型功能
Fast Online Object Tracking and Segmentation,[论文地址](https://arxiv.org/abs/1812.05050), [官方代码](https://github.com/foolwood/SiamMask)
#### Results
|                           <sub>Tracker</sub>                           |      <sub>VOT2016</br>EAO /  A / R</sub>  |      <sub>VOT2016</br>EAO /  A / R</sub>  |
|:----------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| 论文SiamMask | <sub>**0.433**/**0.639**/**0.214**</sub> | 56 FPS | 
| 推理SiamMask | <sub>**0.244**/**0.629**/**0.606**</sub> | 61 FPS | 
#### 代码及路径解释
```
.
└── SiamMask_tf
    ├── config 存放配置文件
    │   ├── config.json
    ├── data 用于制作训练集，验证集，测试集的标签（参考官方代码）
    │   ├── VOT2016
    │   ├── VOT2016.json
    │   ├── create_json.py
    │   ├── get_test_data.sh
    ├── tools 推理及其后处理
    │   ├── eval.py
    │   ├── test_npu_inference.py
    ├── utils 公共方法（参考官方代码）
    │   ├── pysot
    │   ├── pyvotkit
    │   ├── ...
    ├── README.md
    ├── convert_image2bin_inference.sh 推理脚本
    ├── requirements_all.txt
    └── inference.sh 推理启动脚本
```

#### 2、原始模型
参考[训练模型代码](../siammask_tf/)
训练生成ckpt模型，to_pb.py 转pb模型
- 百度网盘：链接：https://pan.baidu.com/s/1M35mcoZvysxRoZdLTqnWrQ   提取码：qbvi


#### 3、转om模型
- obs链接pb：https://siammask-zx.obs.cn-north-4.myhuaweicloud.com/siammask.pb  (如果obs失效，请使用百度网盘地址)
- obs链接om：https://siammask-zx.obs.cn-north-4.myhuaweicloud.com/siammask.om
- 百度网盘：链接：https://pan.baidu.com/s/1M35mcoZvysxRoZdLTqnWrQ   提取码：qbvi 
- ATC转换命令：
```
/home/HwHiAiUser/Ascend/ascend-toolkit/20.1.rc1/atc/bin/atc --input_shape="template:1,127,127,3;search:1,255,255,3" --check_report=/home/HwHiAiUser/modelzoo/siammask/device/network_analysis.report --input_format=NHWC --output="/home/HwHiAiUser/modelzoo/siammask/device/siammask" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/mySiamMask/ATC_siammask_tf/model/siammask.pb" 
```
#### 4、将转换的om文件放在model文件夹
- 百度网盘：链接：https://pan.baidu.com/s/1M35mcoZvysxRoZdLTqnWrQ   提取码：qbvi
```
cd model
wget https://siammask-zx.obs.cn-north-4.myhuaweicloud.com/siammask.om
```

#### 5、编译msame推理工具
- 项目根目录下载tools,版本号d58cd31，参考https://gitee.com/ascend/tools/,  编译出msame推理工具
```
git clone https://gitee.com/ascend/tools.git
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd tools/msame/
./build.sh g++ ./
```



#### 6、性能测试
回到项目根目录，使用msame推理工具，参考如下命令，发起推理性能测试：
./tools/msame/out/msame --model model/siammask.om --output output/ --loop 100
```
******************************
Test Start!
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] malloc buffer for mem , require size is 31197184
[INFO] malloc buffer for weight,  require size is 32505856
[INFO] load model model/siammask.om success
[INFO] create model description success
[INFO] create model output success
[INFO] model execute success
Inference time: 16.333ms
output//20210202_172031
[INFO] output data success
Inference average time: 16.333000 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```
平均推理性能16.333ms


#### 7、test Data 
- 使用get_test_data.sh下载数据集,只使用VOT2016即可（VOT2016包含VOT2016文件夹和VOT2016.json）
- 也可以直接删除./data目录，再根据下方的[OBS链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=hCBKrEZG4sAIrJfXQrXfGsSoRN+g0YNZ9iH1jnDDfzcjjof2ynnXfJ4VYPm/eYo2//6r7VsH19NSgmSA9MrPF0ksNEr7EuSmWmpbDCUVSrk4RKEBOjF9ZJvuMxj+143L6oK3tUDGvIA+fVnrC28m+V2kIEWlvAGH7YKXQ/s4cCMagzvT3OpVDWM2AD7vRsQFlwidqaJTXTkFmCuaCaws1jQzaLzWzP0hMpNCcRvOMfJbQTjwxMBKPBmvVdaOj+/0+m1lMJladvzQKBRkh2hyP0osRrYeKY/UdXdwpo+32rm3ixwEJs0vg08Q1seNgNlmbebWB4zVCZu+F+2MgPJab5xTWCZTfMjn+E1qzMsJ4WHm1z4tKgKA4bB/6/pf/f5cNsp2I4ejA/qRZiwUOVxn8Gn0K/HuQ4cMlWdSfj23I+AE2+h6pH1dhZtKwXGCGlFzVbCrKBI1bp6+rCoGFW5flGd7+o/9U6gOLTXXT84PmyVCw/E0xq+ZgdbbvNlYunqdW1ljwm2gYNQT1XmDTvnWiMzvVr7XWzMPnP90tIjXFI4=)下载数据集,提取密码siamma, 并将下载好数据集各自解压到 ./data/目录下，以下是数据解压后的一个示例。
- 百度网盘：链接：https://pan.baidu.com/s/1M35mcoZvysxRoZdLTqnWrQ   提取码：qbvi
```
cd data
wget https://siammask-zx.obs.cn-north-4.myhuaweicloud.com/data/VOT2016.zip
wget https://siammask-zx.obs.cn-north-4.myhuaweicloud.com/data/VOT2016.json
unzip VOT2016.zip
```


#### 8、执行推理和精度计算的python脚本
- 执行推理脚本
```
sudo pip3.7 install Cython
sudo pip3.7 install tqdm
sudo pip3.7 install opencv-python
sudo pip3.7 install colorama
sudo pip3.7 install numba

bash make.sh
bash inference.sh
```
等待时间较久(12小时)，最后精度EAO为0.244