### 概要

网络名/用途：DPN / 图像分类
论文链接/名称：https://arxiv.org/abs/1707.01629 / Dual Path Networks
源码链接：https://github.com/Stick-To/DPN-tensorflow
数据集名称/归档位置：cifar10/ 10.136.165.4:/turingDataset/cifar10
loss+perf_gpu.txt、ckpt_gpu归档路径：10.136.165.4:/turingDataset/GPU/CarPeting_TF_DPN 
ckpt_npu归档路径：10.136.165.4:/turingDataset/results/CarPeting_TF_DPN 

### 环境

- python3.6
- tensorflow>=1.12

### 修改代码

在 `test.py` 中将以下部分：

```
import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
```

替换为：

```
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import time
```

修改配置：

```
num_train = 640

epochs = 1
```

在这句后：

```
testnet = net.DPN(config, data_shape, num_classes, weight_decay, keep_prob, 'channels_last')
```

添加：

```
tf.io.write_graph(testnet.sess.graph, 'output', 'graph.pbtxt', as_text=True)
```

将以下部分：

```
    # train one epoch
    for iter in range(num_train//train_batch_size):
        # get and preprocess image
        images, labels = train_gen.next()
        images = images - mean
        # train_one_batch also can accept your own session
        loss, acc = testnet.train_one_batch(images, labels, lr)
        train_acc.append(acc)
        train_loss.append(loss)
        sys.stdout.write("\r>> train "+str(iter+1)+'/'+str(num_train//train_batch_size)+' loss '+str(loss)+' acc '+str(acc))
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    sys.stdout.write("\n")
    print('>> epoch', epoch, 'train mean loss', mean_train_acc, 'train mean acc', mean_train_acc)
```

替换为：

```
    begin_epoch = time.time()
    # train one epoch
    for iter in range(num_train//train_batch_size):
        # get and preprocess image
        images, labels = train_gen.next()
        images = images - mean
        begin = time.time()
        # train_one_batch also can accept your own session
        loss, acc = testnet.train_one_batch(images, labels, lr)
        end = time.time()
        train_acc.append(acc)
        train_loss.append(loss)
        print('step:', str(iter + 1) + '/' + str(num_train // train_batch_size), 'time:', '%.3fs' % (end - begin), 'loss:', loss, 'acc:', acc)
        sys.stdout.flush()
    end_epoch = time.time()
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    print('>> epoch:', epoch + 1, 'time:', '%.3fs' % (end_epoch - begin_epoch), 'loss:', mean_train_loss, 'acc:', mean_train_acc)
    sys.stdout.flush()
    tf.train.Saver().save(testnet.sess, "output/model.ckpt-" + str(epoch))
```


将以下部分注释：

```
    # validate one epoch
    for iter in range(num_test//test_batch_size):
        # get and preprocess image
        images, labels = test_gen.next()
        images = images - mean
        # validate_one_batch also can accept your own session
        logit, acc = testnet.validate_one_batch(images, labels)
        test_acc.append(acc)
        sys.stdout.write("\r>> test "+str(iter+1)+'/'+str(num_test//test_batch_size)+' acc '+str(acc))
    mean_val_acc = np.mean(test_acc)
    sys.stdout.write("\n")
    print('>> epoch', epoch, ' test mean acc', mean_val_acc)
```

### 准备数据集

下载数据集 cifar-10-python.tar.gz 到目录 ～/.keras/datasets 中，并重命名为 cifar-10-batches-py.tar.gz

### GPU训练

```
cd $PATH_ROOT
mkdir output
```
新建 `$PATH_ROOT/run_1p.sh` ：
```
#!/bin/bash
rm -rf output/*
python3 test.py
```
执行：

```
sh run_1p.sh > loss+perf_gpu.txt 2>&1
```

训练过程耗时约**8分钟**，loss收敛趋势：

```
-------------------- epoch 0 --------------------
reduce learning rate = 0.0316227766016838 now
step: 1/5 time: 171.016s loss: 21.556992 acc: 0.0546875
step: 2/5 time: 3.687s loss: 21.34834 acc: 0.109375
step: 3/5 time: 2.694s loss: 21.441696 acc: 0.1015625
step: 4/5 time: 2.666s loss: 21.434765 acc: 0.1171875
step: 5/5 time: 2.185s loss: 21.586626 acc: 0.1015625
>> epoch: 1 time: 182.498s loss: 21.473684 acc: 0.096875
```

checkpoint文件在 `$PATH_ROOT/output` 目录

### NPU训练

复制 ``run_npu_1p.sh`` 新建 `$PATH_ROOT/run_1p.sh` 并追加：

```
rm -rf output/*
python3 test.py
```
执行：
```
sh run_1p.sh > loss+perf_npu.txt 2>&1
```

训练过程耗时约**22分钟**，loss收敛趋势：

```
-------------------- epoch 0 --------------------
reduce learning rate = 0.0316227766016838 now
step: 1/5 time: 254.489s loss: 21.537096 acc: 0.1015625
step: 2/5 time: 8.529s loss: 21.44765 acc: 0.078125
step: 3/5 time: 6.489s loss: 21.36703 acc: 0.1015625
step: 4/5 time: 5.917s loss: 21.406965 acc: 0.125
step: 5/5 time: 6.358s loss: 21.510695 acc: 0.1015625
>> epoch: 1 time: 282.309s loss: 21.453888 acc: 0.1015625
```

checkpoint文件在 `$PATH_ROOT/output` 目录