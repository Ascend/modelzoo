# -*-coding:utf8-*-

import os
import sys
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    checkpoint_dir = sys.argv[1]
    step0 = os.path.join(checkpoint_dir, 'model.ckpt-0')
    step1 = os.path.join(checkpoint_dir, 'model.ckpt-1')

    reader0 = tf.train.NewCheckpointReader(filepattern=step0)
    reader1 = tf.train.NewCheckpointReader(filepattern=step1)
    var_to_shape_map0 = reader0.get_variable_to_shape_map()
    var_to_shape_map1 = reader1.get_variable_to_shape_map()
    var_dic0 = {}
    var_dic1 = {}
    for key in sorted(var_to_shape_map0):
        if 'adam' in key.lower():
            continue
        var_dic0[key] = reader0.get_tensor(key)
    for key in sorted(var_to_shape_map1):
        if 'adam' in key.lower():
            continue
        var_dic1[key] = reader1.get_tensor(key)

    for k in var_dic0.keys():
        change = var_dic1[k] - var_dic0[k]
        total = np.sum(np.abs(change))
        print('parameter: ', k, 'change: ', total)
