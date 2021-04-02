from __future__ import print_function
from __future__ import division

import pickle

import numpy as np
import tensorflow as tf


def normalize(norm, x, num_inputs):
    if norm:
        with tf.name_scope('norm'):
            return x / np.sqrt(num_inputs)
    else:
        return x


def build_optimizer(_ptmzr_argv, loss):
    _ptmzr = _ptmzr_argv[0]
    if _ptmzr == 'adam':
        _learning_rate, _epsilon = _ptmzr_argv[1:3]
        ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate,
                                       epsilon=_epsilon).minimize(loss)
        log = 'optimizer: %s, learning rate: %g, epsilon: %g' % \
              (_ptmzr, _learning_rate, _epsilon)
    elif _ptmzr == 'adagrad':
        _learning_rate, _initial_accumulator_value = _ptmzr_argv[1:3]
        ptmzr = tf.train.AdagradOptimizer(
            learning_rate=_learning_rate,
            initial_accumulator_value=_initial_accumulator_value).minimize(loss)
        log = 'optimizer: %s, learning rate: %g, init_accumulator_value: %g' % (
            _ptmzr, _learning_rate, _initial_accumulator_value)
    elif _ptmzr == 'ftrl':
        _learning_rate, init_accum, lambda_1, lambda_2 = _ptmzr_argv[1:5]
        ptmzr = tf.train.FtrlOptimizer(
            learning_rate=_learning_rate,
            initial_accumulator_value=init_accum,
            l1_regularization_strength=lambda_1,
            l2_regularization_strength=lambda_2).minimize(loss)
        log = ('optimizer: %s, learning rate: %g, initial accumulator: %g, '
               'l1_regularization: %g, l2_regularization: %g' %
               (_ptmzr, _learning_rate, init_accum, lambda_1, lambda_2))
    else:
        _learning_rate = _ptmzr_argv[1]
        ptmzr = tf.train.GradientDescentOptimizer(
            learning_rate=_learning_rate).minimize(loss)
        log = 'optimizer: %s, learning rate: %g' % (_ptmzr, _learning_rate)
    return ptmzr, log


def activate(act_func, x):
    if act_func == 'tanh':
        return tf.tanh(x)
    elif act_func == 'relu':
        return tf.nn.relu(x)
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
                        _j += 1
                    else:
                        var_map[key] = tf.zeros(shape)
                        _j += 1
                else:
                    print('%s already set' % key)
    return var_map, log


def layer_normalization(_input_tensor, gain, biase, epsilon=1e-5):
    layer_mean, layer_variance = tf.nn.moments(_input_tensor, [1], keep_dims=True)
    layer_norm_input = (_input_tensor-layer_mean)/tf.sqrt(layer_variance + epsilon)
    return layer_norm_input * gain + biase


def split_mask(mask, multi_hot_flags, num_multihot):
    """split original mask into 2 part: one-hot, multi-hot;
    and substitute multi-hot mask with mean value.
    """
    # multi_hot_mask = tf.boolean_mask(mask, multi_hot_flags, axis=1)
    multi_hot_mask = tf.transpose(
        tf.boolean_mask(tf.transpose(mask, [1, 0, 2]),
                        multi_hot_flags),
        [1, 0, 2])

    mul_mask_list = tf.split(multi_hot_mask, num_multihot, axis=1)
    mul_mask_list_proc = []
    for mul_mask in mul_mask_list:
        sum_mul_mask = tf.reduce_sum(mul_mask, 1, keep_dims=True)
        sum_mul_mask = tf.maximum(sum_mul_mask, tf.ones_like(sum_mul_mask))
        mul_mask /= sum_mul_mask
        mul_mask_list_proc.append(mul_mask)
    multi_hot_mask = tf.concat(mul_mask_list_proc, axis=1)
    multi_hot_mask.set_shape((None, sum(multi_hot_flags), None))

    one_hot_flags = [not flag for flag in multi_hot_flags]
    # one_hot_mask = tf.boolean_mask(mask, one_hot_flags, axis=1)
    one_hot_mask = tf.transpose(
        tf.boolean_mask(tf.transpose(mask, [1, 0, 2]),
                        one_hot_flags),
        [1, 0, 2])
    one_hot_mask.set_shape((None, sum(one_hot_flags), None))
    return one_hot_mask, multi_hot_mask


# todo: change var name. this function accepts any flags.(e.g. multi-hot, continuous)
def split_param(model_param, id_hldr, multi_hot_flags):
    """split model_param into 2 part: one-hot params and multi-hot params.
    :param model_param: w [input_dim, 1] or v [input_dim, k]
    :param id_hldr: placeholder with mini-batch feature ids.
    :param multi_hot_flags: list
    :return:
    """
    one_hot_flags = [not flag for flag in multi_hot_flags]
    batch_param = tf.gather(model_param, id_hldr)
    # batch_one_hot_param = tf.boolean_mask(batch_param, one_hot_flags, axis=1)
    # batch_multi_hot_param = tf.boolean_mask(
    #     batch_param, multi_hot_flags, axis=1)
    batch_param = tf.transpose(batch_param, [1, 0, 2])
    batch_one_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, one_hot_flags),
        [1, 0, 2])
    batch_one_hot_param.set_shape((None, sum(one_hot_flags), None))
    batch_multi_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, multi_hot_flags),
        [1, 0, 2])
    batch_multi_hot_param.set_shape((None, sum(multi_hot_flags), None))
    return batch_one_hot_param, batch_multi_hot_param


def sum_multi_hot(batch_multi_hot_param, multi_hot_mask, num_multihot):
    """sum multi-hot features params into a single one.
    :param batch_multi_hot_param: params to be summed up within same field.
    :param multi_hot_mask: list.
    :param num_multihot: multi-hot field number.(assume
           each field has the same length)
    :return:
    """
    param_masked = tf.multiply(batch_multi_hot_param, multi_hot_mask)
    param_list = tf.split(param_masked, num_multihot, axis=1)
    param_reduced_list = []
    for param_ in param_list:
        param_reduced = tf.reduce_sum(param_, axis=1, keep_dims=True)
        param_reduced_list.append(param_reduced)
    return tf.concat(param_reduced_list, axis=1)


def get_field_index(multi_hot_flags):
    """infer field index of each column using `multi_hot_flags`
    Example:
    get_field_index([False,False,True,True,True,False])  # [0,1,2,2,2,3]
    """
    field_indices = []
    cur_field_index = 0
    for i, flag in enumerate(multi_hot_flags):
        field_indices.append(cur_field_index)
        if not flag or (
                flag and (i+1 < len(multi_hot_flags))
                and not multi_hot_flags[i+1]):
            cur_field_index += 1
    return field_indices


def get_field_num(multi_hot_flags, multi_hot_len):
    """ infer field number
    Example:
        get_field_num([False,False,True,True,True,False], 3)  # 4
        get_field_num([False,True,True,True,True,False], 2)  # 4
        get_field_num([False,True,True,True,True,False], 4)  # 3
    """
    one_hot_flags = [not flag for flag in multi_hot_flags]
    one_hot_field_num = sum(one_hot_flags)
    if sum(multi_hot_flags) % multi_hot_len != 0:
        raise ValueError("cannot infer field number. please check input!")
    multi_hot_field_num = sum(multi_hot_flags) // multi_hot_len
    field_num = one_hot_field_num + multi_hot_field_num
    return field_num
