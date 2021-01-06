# coding: UTF-8

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import argparse
import numpy as np

from models.resnet50 import res50_model as ml
from configs.res50_256bs_1p import res50_config

config = res50_config()
tf.reset_default_graph()


def add_placeholder_on_ckpt(input_file):
    save_path = os.path.join("/cache/ckpt_first", 'placeholder',
                             os.path.basename(input_file))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = ml.Model(config, None, None, None, None, None, None)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1,
                                             config.get("height"),
                                             config.get("width"),
                                             3])

        with tf.variable_scope('fp32_vars'):
            model_func = model.get_model_func()
            top_layer = model_func(
                inputs, data_format=config['data_format'],
                training=False,
                conv_initializer=config['conv_init'],
                bn_init_mode=config['bn_init_mode'],
                bn_gamma_initial_value=config['bn_gamma_initial_value'])

        saver = tf.train.Saver(var_list=tf.global_variables(scope='fp32_vars'))
        sess.run(tf.global_variables_initializer())
        saver_to_restore = tf.train.Saver()
        saver_to_restore.restore(sess, input_file)
        saver.save(sess, save_path=save_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(
            save_path))


# weight_path = tf.train.latest_checkpoint(config.get('model_dir'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='the input file name')
    return parser.parse_args(argv)


def main():
    # args = ['model-epoch_17_step_9350_loss_3.8476_lr_0.001875']
    args = None
    args = parse_arguments(args)
    add_placeholder_on_ckpt(args.input_file)


if __name__ == '__main__':
    main()

