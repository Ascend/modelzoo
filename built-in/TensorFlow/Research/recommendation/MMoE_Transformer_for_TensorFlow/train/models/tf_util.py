from __future__ import print_function
from __future__ import division

import pickle

import numpy as np
import tensorflow as tf


def activate(act_func, x):
    if act_func == 'tanh':
        return tf.tanh(x)
    elif act_func == 'relu':
        return tf.nn.relu(x)
    elif act_func == 'softmax':
        return tf.nn.softmax(x)
    else:
        return tf.sigmoid(x)


def init_var_map(_init_argv, vars):
    _init_path = _init_argv[-1]
    if _init_path:
        var_map = pickle.load(open(_init_path, 'rb'))
        log = 'init model from: %s, ' % _init_path
    else:
        var_map = {}
        log = 'random init, '

        _init_method = _init_argv[0]
        if _init_method == 'normal':
            _mean, _stddev, _seeds = _init_argv[1:-1]
            log += 'init method: %s(mean=%g, stddev=%g), seeds: %s\n' % (
                _init_method, _mean, _stddev, str(_seeds))
            _j = 0
            for _i in range(len(vars)):
                key, shape, action = vars[_i]
                if key not in var_map.keys():
                    if action == 'random' or action == "normal":
                        print('%s normal(mean=%g, stddev=%g) random init' %
                              (key, _mean, _stddev))
                        var_map[key] = tf.random_normal(shape, _mean, _stddev,
                                                        seed=_seeds[_j % 10])
                        # var_map[key] = tf.random_normal(shape, _mean, _stddev)
                        _j += 1
                    else:
                        var_map[key] = tf.zeros(shape)
                else:
                    print('%s already set' % key)
        else:
            _min_val, _max_val, _seeds = _init_argv[1:-1]
            log += 'init method: %s(minval=%g, maxval=%g), seeds: %s\n' % (
                _init_method, _min_val, _max_val, str(_seeds))
            _j = 0
            for _i in range(len(vars)):
                key, shape, action = vars[_i]
                if key not in var_map.keys():
                    if action == 'random' or action == 'uniform':
                        print('%s uniform random init, ' % key,
                              "(minval=%g, maxval=%g)\nseeds: %s" % (
                                  _min_val, _max_val, str(_seeds)))
                        var_map[key] = tf.random_uniform(
                            shape, _min_val, _max_val,
                            seed=_seeds[_j % len(_seeds)])
                        # var_map[key] = tf.random_uniform(
                        #     shape, _min_val, _max_val)
                        _j += 1
                    else:
                        var_map[key] = tf.zeros(shape)
                        _j += 1
                else:
                    print('%s already set' % key)
    return var_map, log