# 概述
## 来源
Stein variational gradient descent （SVGD）是由 Liu Qiang 等提出的一种近似推断算法。

## 原理

SVGD是一种结合了 Variational Inference 和 Markov Chain Monte Carlo 的近似推断方法，保有了MCMC的渐进一致性，又在运行速度上具有优势。

# 环境依赖
* Tensorflow 1.15.0

# 文件结构
```
SVGD
└─ 
  ├─ readme.md
  ├─ references 论文作者实现
    ├─ gaussian_mixture.py
    ├─ svgd.py
  ├─ results 输出结果
    ├─ 1_gaussian_mixture
    ├─ 2_bayesian_classification
  ├─ tests 对比测试脚本
    ├─ test_gaussian_mixture.py
    ├─ test_median.py
    ├─ test_svgd_kernel.py
  ├─ 1_gaussian_mixture.py
  ├─ 2_bayesian_classification.py
  ├─ optimizer.py SVGD的tensorflow实现
```

# 使用方法
* 定义网络，获取 `gradients` 和 `variables`
```python
def network():
    '''
    Define target density and return gradients and variables. 
    '''
    return gradients, variables
```

* 定义梯度下降优化器
```python
def make_gradient_optimizer():
    return tf.train.GradientDescentOptimizer(learning_rate=0.01)
```

* 用 `network()` 构建网络（粒子）并将所有 `gradients` 和 `variables` 放在 `grads_list` 和 `vars_list` 中
* 得到 SVGD 优化器
```python
optimizer = SVGD(grads_list, vars_list, make_gradient_optimizer)
```
* 训练阶段，通过 `optimizer.update_op` 更新
```python
sess = tf.Session()
sess.run(optimizer.update_op, feed_dict={X: x, Y: y})
```

# 运行方法
验证实验
```
python 1_gaussian_mixture.py
python 2_bayesian_classification.py
```

SVGD算法
```
python optimizer.py
```
