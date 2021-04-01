



# WaveGlow网络迁移

```sh
开源网址：https://github.com/weixsong/WaveGlow
```

##### 该网络基于20210302主线newest版本迁移。



### 数据集：

LJSpeech-1.1

预处理后已归档在10.136.165.4服务器/turingDataset/waveglow/LJSpeech-1.1下



### 训练步骤：

#### 1、数据集预处理（参考开源网络readme）：

按照以下命令通过**preprocess_data.py**处理数据： 

```shell
python3 preprocess_data.py --wave_dir=./LJSpeech-1.1/wavs/ --mel_dir=./LJSpeech-1.1/mels/ --data_dir=./LJSpeech-1.1/
```

#### 2、训练模型

##### 1、模型参数在文件params.py中

params.py文件中gen_file参数需修改，示例如下：

```shell
    # train
    lr=0.001,
    train_steps=60,
    save_model_every=20,
    gen_test_wave_every=20,
    gen_file='/data/WYF/move/WaveGlow-master/LJSpeech-1.1/mels/LJ001-0001.mel',
    logdir_root='./logdir',
    decay_steps=30,
    sigma=0.707,
    
    
    原始参数值：
    train_steps=1000000,
    save_model_every=4000,
    gen_test_wave_every=10000,
    gen_file='./LJSpeech-1.1/mels/LJ001-0001.mel',
    logdir_root='./logdir',
    decay_steps=50000,
    sigma=0.707,
    
```

##### 2、需要手工迁移部分

1、meta文件开关

```shell
在train.py文件中需将write_meta_graph=Flase设置成True，如下：
def save(saver, sess, logdir, step, write_meta_graph=True):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end='')
```

在train.py和inference.py文件中需添加

```shell
    # 手工迁移
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
    sess = tf.Session(config=config)
    #
```

##### 3、训练命令：

```shell
python3 train.py --filelist=./LJSpeech-1.1/train.scp --wave_dir=./LJSpeech-1.1/wavs/ --lc_dir=./LJSpeech-1.1/mels/

参数说明：
--filelist：预处理后train.scp路径
--wave_dir：wavs路径
--lc_dir：mel图路径
```

##### 4、结果检查：

```shell
step 34 - loss = -1.634, lr=0.00095000, time cost=0.200091
step 35 - loss = -1.692, lr=0.00095000, time cost=0.200156
step 36 - loss = -1.823, lr=0.00095000, time cost=0.200260
step 37 - loss = -1.626, lr=0.00095000, time cost=0.200006
step 38 - loss = -1.703, lr=0.00095000, time cost=0.199976
step 39 - loss = -0.891, lr=0.00095000, time cost=0.200091
step 40 - loss = -1.853, lr=0.00095000, time cost=0.200032
Storing checkpoint to ./logdir/waveglow ...
 Done.
Updated wav file at ./logdir/waveglow/wave/00000040.wav
step 41 - loss = -1.739, lr=0.00095000, time cost=0.202939
step 42 - loss = -1.705, lr=0.00095000, time cost=0.200773
step 43 - loss = -0.870, lr=0.00095000, time cost=0.200487
step 44 - loss = -1.667, lr=0.00095000, time cost=0.200710
step 45 - loss = -1.701, lr=0.00095000, time cost=0.200496
step 46 - loss = -1.732, lr=0.00095000, time cost=0.200982
step 47 - loss = -1.656, lr=0.00095000, time cost=0.200531
step 48 - loss = -1.746, lr=0.00095000, time cost=0.200403
step 49 - loss = -1.778, lr=0.00095000, time cost=0.200649
step 50 - loss = -1.676, lr=0.00095000, time cost=0.200465
step 51 - loss = -1.803, lr=0.00095000, time cost=0.200490
step 52 - loss = -1.780, lr=0.00095000, time cost=0.200364
step 53 - loss = -1.847, lr=0.00095000, time cost=0.200319
step 54 - loss = -1.812, lr=0.00095000, time cost=0.200273
step 55 - loss = -1.828, lr=0.00095000, time cost=0.200288
step 56 - loss = -2.059, lr=0.00095000, time cost=0.200378
step 57 - loss = -1.744, lr=0.00095000, time cost=0.200358
step 58 - loss = -2.083, lr=0.00095000, time cost=0.200335
step 59 - loss = -2.253, lr=0.00095000, time cost=0.200402
```



#### 3、在线推理

```shell
输入节点
Tensor("lc_infer:0", shape=(1, ?, 80), dtype=float32)
输出节点
Tensor("Waveglow/Reshape_2:0", shape=(?, ?, 1), dtype=float32)
```

##### ckpt转pb命令：

```shell
python3 convert_ckpt_to_pb.py
```

##### 数据集准备：

```shell
#原始音频文件(需要做预处理请执行)
#如果使用原始音频执行预处理，则不需要再拷贝bin文件
mkdir -p datasets/wavs_5
scp -r autotest@10.136.165.4:/turingDataset/waveglow/LJSpeech-1.1/wavs_5 ./LJSpeech-1.1

#转换后的bin文件(直接拷贝用于推理)
mkdir -p datasets/wavs_5
mkdir result_wav
scp -r autotest@10.136.165.4:/turingDataset/waveglow/LJSpeech-1.1/binfile/binfile_5 ./datasets
```

##### 在线推理命令：

```shell
python3 wave_online_inference.py
```

##### 结果校验：

```shell
在线推理：
结果文件存放在 ./perform_static_dev_0_chn_0.txt文件中
```

```shell
性能示例:
average infer time: 39638.647 ms 0.025 fps/s
```

