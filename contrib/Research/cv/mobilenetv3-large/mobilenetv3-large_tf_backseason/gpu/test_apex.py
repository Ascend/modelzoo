import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

in_p = tf.placeholder(tf.float32, [None, 128, 128, 3])
net = K.layers.Conv2D(3, 32, 2, 'same', activation=K.activations.elu)(in_p)
net = K.layers.Conv2D(3, 64, 2, 'same', activation=K.activations.elu)(net)
net = K.layers.Conv2D(3, 128, 2, 'same', activation=K.activations.elu)(net)
net = K.layers.Conv2D(3, 256, 2, 'same', activation=K.activations.elu)(net)
net_out = tf.reduce_mean(net, axis=[1,2,3])

loss_op = tf.reduce_mean(tf.abs(net_out - tf.ones_like(net_out)))

optim = tf.train.AdamOptimizer()
optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim)
optim_op = optim.minimize(loss_op)

s = tf.compat.v1.Session()
s.run(tf.compat.v1.global_variables_initializer())

for i in range(1000):
    x = np.random.normal(0, 1, [10, 128, 128, 3])
    loss, _ = s.run([loss_op, optim_op], {in_p: x})
    print(loss)
