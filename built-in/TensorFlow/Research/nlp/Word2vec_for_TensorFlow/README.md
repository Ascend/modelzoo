



# Word2vec网络迁移

```sh
开源网址：https://github.com/Deermini/word2vec-tensorflow
```

概述：https://zhuanlan.zhihu.com/p/28979653 

##### 该网络基于20210309主线newest版本迁移。



### 数据集：

cnews

下载地址：https://pan.baidu.com/s/1hugrfRu （验证码：qfud ）

归档在10.136.165.4服务器/turingDataset/cnews下



### 训练步骤：

#### 1、训练模型

##### 1、word2vec_chinese.py文件中需手工修改部分：

```shell
1、开源网络未增加性能打点，修改如下：
import time

with tf.Session(graph=graph, config=npu_session_config_init()) as session:
    init.run()
    print('Initialized')
    average_loss = 0
    duration = 0	#add
    for step in xrange(num_steps):
        (batch_inputs, batch_labels) = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        start_time = time.time()	#add
        (_, loss_val) = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        duration += (time.time() - start_time)	#add
        if ((step % 2000) == 0):
            if (step > 0):
                average_loss /= 2000
#            print('Average loss at step ', step, ': ', average_loss)
            print('step = ', step, 'loss = {:3f}, time cost = {:4f}'.format(average_loss, duration))	#add
            average_loss = 0
            duration = 0	#add
            
2、原始网络在windows上训练，linux上simsun.ttc文件需配置：
原始为：
font = FontProperties(fname='c:\\windows\\fonts\\simsun.ttc', size=14)
修改为：
font = FontProperties(fname='./SIMSUN.TTC', size=14)

3、原始数据集为斗破苍穹，改为cnews，验证词需修改
原始为：
valid_word = ['萧炎', '灵魂', '火焰', '萧薰儿', '药老', '天阶', '云岚宗', '乌坦城', '惊诧']
修改为：
valid_word = ['城市', '记者', '体育', '教练', '足球', '赛季', '奥运会', '丑闻', '足协']
```

##### 2、训练命令：

```shell
bash run_npu_1p.sh
```

##### 3、结果检查：

```shell
step =  12000 loss = 15.024554, time cost = 38.142089
step =  14000 loss = 12.462806, time cost = 38.360088
step =  16000 loss = 11.353894, time cost = 38.454652
step =  18000 loss = 10.704374, time cost = 37.738785
step =  20000 loss = 7.996528, time cost = 38.367597
Nearest to 城市: 城市, 部分, 外刊, 农大, 针对, 制片, 金, 33,
Nearest to 记者: 记者, 预定, 澳大利亚, 话, UV, 晒, Because, 邻,
Nearest to 体育: 体育, 牌, 项,  , 若, 隐蔽性, 诠释, 脱颖而出,
Nearest to 教练: 教练, 光年, 常, 丢, 插花, 条, 片场, 文本,
Nearest to 足球: 足球, 诉讼, 戴欣明, 配, 杨, 方恒, 势头, 成绩单,
Nearest to 赛季: 赛季, 戴欣明, 纠正, 经济, ①, 共同努力, 主力,  ,
Nearest to 奥运会: 奥运会, 乌龙, 布雷克, 到手, 偏暖, 前方, 四五年, 调价,
Nearest to 丑闻: 丑闻, 大米, 令人欣慰, 紫, 外国人, 重视, 项链, 3.15,
Nearest to 足协: 足协, 另外, 遏制, 农民工, 世上, 由, 米歇尔, 时有,
step =  22000 loss = 8.202206, time cost = 43.102700
step =  24000 loss = 8.278960, time cost = 43.722719
step =  26000 loss = 7.141563, time cost = 43.817739
step =  28000 loss = 6.660999, time cost = 43.942462
```

##### 4、NPU-->GPU loss值对比

| step  | GPU        | NPU        |
| ----- | ---------- | ---------- |
| 0     | 289.446472 | 290.149902 |
| 2000  | 111.120605 | 111.780381 |
| 4000  | 49.23201   | 49.497835  |
| 6000  | 37.828253  | 37.7693    |
| 8000  | 26.121089  | 25.779517  |
| 10000 | 20.382499  | 20.385034  |
| 12000 | 15.544454  | 15.206445  |
| 14000 | 12.291553  | 12.278699  |
| 16000 | 11.221707  | 11.207923  |
| 18000 | 10.685602  | 10.69174   |
| 20000 | 8.177321   | 8.026703   |
| 22000 | 7.932244   | 8.074527   |
| 24000 | 8.157113   | 8.169231   |
| 26000 | 7.277833   | 7.160124   |
| 28000 | 6.82544    | 6.687464   |

![1615290838186](D:\training_shop\03-code\CarPeting_Word2vec_TF\1615290838186.png)