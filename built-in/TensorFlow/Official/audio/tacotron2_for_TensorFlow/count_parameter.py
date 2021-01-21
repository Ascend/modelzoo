# -*-coding:utf8-*-

import os
import sys
import tensorflow as tf


if __name__ == '__main__':
    checkpoint_dir = sys.argv[1]
    latest_ckp = checkpoint_dir
    reader = tf.train.NewCheckpointReader(filepattern=latest_ckp)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_dic = {}
    total_params = 0
    for key in sorted(var_to_shape_map):
        var_dic[key] = reader.get_tensor(key)
        # print(key, var_dic[key].shape)
        if 'adam' not in key:
            print(key)
            total_params += var_dic[key].size
    print(total_params)
