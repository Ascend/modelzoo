# DCGAN
原始模型参考[github链接](https://github.com/carpedm20/DCGAN-tensorflow),迁移训练代码到NPU

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- 其他依赖参考requirements.txt
- 数据集，MNIST

## 准备数据
目前仅调测了MNIST数据集，下载数据集至./data/mnist路径，用gzip -d命令解压数据集

## NPU训练
在NPU上面，启动训练，使用下面的命令:
```
python3.7 main_npu.py --dataset mnist --input_height=28 --output_height=28 --train
```
或者直接执行shell:
```
bash train_testcase.sh
```

### TotalLoss趋势比对（NPU vs GPU）
数据集和超参相同时:
```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```
10w个Step，NPU大概花费10小时，TotalLoss收敛趋势基本一致 :\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0114/232451_0023bcbd_8432352.png "屏幕截图.png")

蓝色是NPU，红色是GPU.

### 精度评估
首先确保安装依赖:
```
apt-get install zip
pip3.7 install Polygon3
```
 - 注意需根据实际python环境编辑"lanms/Makefile"文件， 示例**python3.7-config**:
```
CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3.7-config --cflags)
LDFLAGS = $(shell python3.7-config --ldflags)
```

等训练10w个step结束之后，可以使用eval.sh来评估模型的精度，使用的icdar2015测试集：
```
bash eval.sh
```
Details in eval.sh：
```
export output_dir=./output
export ckpt_dir=./checkpoint/
export test_data=./ocr/ch4_test_images

mkdir  ${output_dir}
rm -rf ${output_dir}/*

python3.7 eval.py \
--test_data_path=${test_data} \
--checkpoint_path=${ckpt_dir} \
--output_dir=${output_dir}

cd ${output_dir}
zip results.zip res_img_*.txt
cd ../

python3.7 evaluation/script.py -g=./evaluation/gt.zip -s=${output_dir}/results.zip
```