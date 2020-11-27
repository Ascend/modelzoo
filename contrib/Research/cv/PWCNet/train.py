from __future__ import absolute_import, division, print_function
import cv2
import tensorflow as tf
import numpy as np
import tensorflow as tf
import npu_bridge # 导入TFAdapter插件库
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# config = tf.ConfigProto()
# custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
# custom_op.name = "NpuOptimizer"
# custom_op.parameter_map["use_off_line"].b = True # 必须显式开启，在昇腾AI处理器执行训练
# config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
#
#
# # 首先，创建一个TensorFlow常量，并赋值2
# const = tf.constant(2.0, name='const')
# # 两种方式创建变量b和c
# # 使变量b可以接收任意值。TensorFlow中接收值的方式为占位符(placeholder)，通过tf.placeholder()创建。
# b = tf.placeholder(tf.float32, [None, 1], name='b')
# #使用tf.Variable()定义变量，值可变。
# c = tf.Variable(1.0, dtype=tf.float32, name='c')
#
# # 创建operation
# d = tf.add(b, c, name='d')
# e = tf.add(c, const, name='e')
# a = tf.multiply(d, e, name='a')
#
# # Tensorflow 的变量必须先初始化，然后才有值
# # 添加用于初始化变量的节点
# init_op = tf.global_variables_initializer()
# # 运行graph需要先调用tf.Session()函数创建一个会话(session)。session就是我们与graph交互的handle。
# # session
#
#
# with tf.Session(config=config) as sess:
#     # 2. 运行init operation
#     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start')
#     sess.run(init_op)
#     print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end')
#     # tensorflow里对于暂时不进行赋值的元素有一个称呼叫占位符
#     # feed_dict就是用来赋值的，格式为字典型
#     a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
#     print("Variable a is {}".format(a_out))
#
# print("----------")
# #np.arange(0, 10)生成一维数组[0 1 2 3 4 5 6 7 8 9]
# print(np.arange(0, 10))
# #np.newaxis在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
# print(np.arange(0, 10)[:, np.newaxis])

import moxing as mox
import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(os.listdir())

# shutil.rmtree('../../FlyingChairs_subset')
os.makedirs('./FlyingChairs_subset/')
mox.file.copy_parallel('obs://pwcnn/content/FlyingChairs_subset', './FlyingChairs_subset')


"""
pwcnet_train.ipynb

PWC-Net model training.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Tensorboard:
    [win] tensorboard --logdir=E:\\repos\\tf-optflow\\tfoptflow\\pwcnet-lg-6-2-cyclic-chairsthingsmix
    [ubu] tensorboard --logdir=/media/EDrive/repos/tf-optflow/tfoptflow/pwcnet-lg-6-2-cyclic-chairsthingsmix
"""

import sys
from copy import deepcopy

from dataset_base import _DEFAULT_DS_TRAIN_OPTIONS
from dataset_flyingchairs import FlyingChairsDataset
from dataset_flyingthings3d import FlyingThings3DHalfResDataset
from dataset_mixer import MixedDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TRAIN_OPTIONS

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# TODO: You MUST set dataset_root to the correct path on your machine!
if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = './'
_FLYINGCHAIRS_ROOT = _DATASET_ROOT + 'FlyingChairs_subset'
_FLYINGTHINGS3DHALFRES_ROOT = _DATASET_ROOT + 'FlyingThings3D_HalfRes'
    
# TODO: You MUST adjust the settings below based on the number of GPU(s) used for training
# Set controller device and devices
# A one-gpu setup would be something like controller='/device:GPU:0' and gpu_devices=['/device:GPU:0']
# Here, we use a dual-GPU setup, as shown below
gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

# TODO: You MUST adjust this setting below based on the amount of memory on your GPU(s)
# Batch size
batch_size = 8


# TODO: You MUST set the batch size based on the capabilities of your GPU(s) 
#  Load train dataset
ds_opts = deepcopy(_DEFAULT_DS_TRAIN_OPTIONS)
ds_opts['in_memory'] = False                          # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'                         # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices) # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (256, 448)                  # Crop to a smaller input size
ds1 = FlyingChairsDataset(mode='train_with_val', ds_root=_FLYINGCHAIRS_ROOT, options=ds_opts)
ds_opts['type'] = 'into_future'
# ds2 = FlyingThings3DHalfResDataset(mode='train_with_val', ds_root=_FLYINGTHINGS3DHALFRES_ROOT, options=ds_opts)
ds = MixedDataset(mode='train_with_val', datasets=[ds1], options=ds_opts)


# Display dataset configuration
ds.print_config()


# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TRAIN_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_dir'] = './pwcnet-lg-6-2-cyclic-chairsthingsmix/'
nn_opts['batch_size'] = ds_opts['batch_size']
nn_opts['x_shape'] = [2, ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 3]
nn_opts['y_shape'] = [ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 2]
nn_opts['use_tf_data'] = True # Use tf.data reader
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# Use the PWC-Net-large model in quarter-resolution mode
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2


# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'cyclic'
nn_opts['cyclic_lr_max'] = 5e-04 # Anything higher will generate NaNs
nn_opts['cyclic_lr_base'] = 1e-05
nn_opts['cyclic_lr_stepsize'] = 20000
nn_opts['max_steps'] = 200000
nn_opts['display_step'] = 1
# Below,we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['cyclic_lr_stepsize'] /= len(gpu_devices)
nn_opts['max_steps'] /= len(gpu_devices)
nn_opts['cyclic_lr_stepsize'] = int(nn_opts['cyclic_lr_stepsize'] / (float(ds_opts['batch_size']) / 8))
nn_opts['max_steps'] = int(nn_opts['max_steps'] / (float(ds_opts['batch_size']) / 8))


# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()


# Train the model
nn.train()

