# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS


INF = 10000.0


def DT_FLOAT():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def NP_FLOAT():
  return np.float16 if FLAGS.use_fp16 else np.float32

def DT_INT():
  return tf.int32
