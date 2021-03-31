## 推理

推理精度为 0.899

推理和训练差异分析：

离线推理使用全量的测试集进行验证，且测试集未出现在训练过程中。 

训练过程中使用同样的测试集进行验证，但是在每次验证中，会随机从测试集中采样batch_size个样本进行验证，不能保证全部测试集都进行了验证。
#### 1、转换后的Bin数据集

链接：https://pan.baidu.com/s/1CbgNLx7rmKTinoResZ7pCg 

提取码：fxz1 

#### 2、转换Bin文件的代码

调用datagen.py中的_test_aux_generator函数

#### 3、ckpt转pb代码

ckpt2pb.py 

需要指定checkpoint的地址和输出pb文件的位置。

#### 4、推理结果

链接：https://pan.baidu.com/s/1ngflIObMK8j59en3na1xBw 

提取码：y388 


#### 5、模型后处理

tuili.py

需要指定Bin文件目录和推理后的结果目录

