### 概要

网络名/用途：Bi-LSTM / 命名实体识别
论文链接/名称：无
源码链接：https://github.com/macanv/BERT-BiLSTM-CRF-NER
数据集名称/归档位置：ChineseNER / 10.136.165.4:/turingDataset/CarPeting_TF_BiLSTM_CRF
loss+perf_gpu.txt、ckpt_gpu归档路径：10.136.165.4:/turingDataset/GPU/CarPeting_TF_BiLSTM_CRF
ckpt_npu归档路径：10.136.165.4:/turingDataset/results/CarPeting_TF_BiLSTM_CRF

### 环境

- python3.6
- tensorflow>=1.12
- termcolor>=1.1
- GPUtil>=1.3.0
- pyzmq >= 17.1.0

### 修改代码

在 `bert_base/train/bert_lstm_ner.py` 中将以下部分：

```
        # early stop hook
        ...
        ...
        ...
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping_hook])
```

替换为：

```
        import time
        class MyHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = 0
                self._loss_tensor = tf.get_default_graph().get_tensor_by_name('crf_loss/Mean:0')

            def after_create_session(self, session, coord):
                pass

            def before_run(self, run_context):
                self._begin = time.time()
                self._step += 1
                return tf.train.SessionRunArgs({'loss': self._loss_tensor})

            def after_run(self, run_context, run_values):
                self._end = time.time()
                cost = self._end - self._begin
                loss = run_values.results['loss']
                print('step: %d, cost time: %.3fs, losses: %.3f' % (self._step, cost, loss))

            def end(self, session):
                pass

        myHook = MyHook()

        stepHook = tf.train.StopAtStepHook(5)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[myHook, stepHook])
```

并在以下函数开头：

```
    def model_fn(features, labels, mode, params):
```

添加以下部分：

```
        if mode != tf.estimator.ModeKeys.TRAIN:
            exit(0)
```

### 安装

```
cd $PATH_ROOT
python3 setup.py install
```

### 准备数据集

创建目录

```
cd $PATH_ROOT
mkdir NERdata
cd NERdata
```

下载数据集 https://github.com/zjy-ucas/ChineseNER 

下载 `data` 目录中的 `example.train`、`example.dev`、`example.test` 

重命名为 `train.txt`、`dev.txt`、`test.txt`

### 预训练模型

下载Google BERT 预训练模型：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip 

解压到项目中重命名为 `model`

### GPU训练

```
cd $PATH_ROOT
mkdir output
```
新建 `$PATH_ROOT/run_1p.sh` ：
```
#!/bin/bash
rm -rf output/*
python3 run.py -do_predict -data_dir NERdata -vocab_file model/vocab.txt -bert_config model/bert_config.json -init_checkpoint model/bert_model.ckpt -output_dir output -batch_size 32 -learning_rate 2e-5 -num_train_epochs 1 -device_map 5
```
-do_train, -do_eval, -do_predict 带上标签代表不进行那个阶段，必须选择一个阶段进行，train和eval阶段要同时进行

执行：

```
sh run_1p.sh > loss+perf_gpu.txt 2>&1
```

训练过程耗时约**2分钟**，loss收敛趋势：

```
step: 1, cost time: 7.391s, losses: 132.724
step: 2, cost time: 6.387s, losses: 139.579
step: 3, cost time: 0.534s, losses: 137.538
step: 4, cost time: 0.490s, losses: 123.286
step: 5, cost time: 0.487s, losses: 124.577
```

checkpoint文件在 `$PATH_ROOT/output` 目录

### NPU训练

复制 ``run_npu_1p.sh`` 新建 `$PATH_ROOT/run_1p.sh` 并追加：

```
rm -rf output/*
python3 run.py -do_predict -data_dir NERdata -vocab_file model/vocab.txt -bert_config model/bert_config.json -init_checkpoint model/bert_model.ckpt -output_dir output -batch_size 32 -learning_rate 2e-5 -num_train_epochs 1
```
执行：
```
sh run_1p.sh > loss+perf_npu.txt 2>&1
```

训练过程耗时约**15分钟**，loss收敛趋势：

```
step: 1, cost time: 164.408s, losses: 121.096
step: 2, cost time: 157.066s, losses: 111.109
step: 3, cost time: 79.547s, losses: 99.554
step: 4, cost time: 79.599s, losses: 93.504
step: 5, cost time: 79.854s, losses: 102.702
```

checkpoint文件在 `$PATH_ROOT/output` 目录