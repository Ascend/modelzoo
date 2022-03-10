#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
# import keras
import tensorflow.python.keras as keras
import argparse
import os
import tf_models
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda, Permute
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Reshape
from tensorflow.python.keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.normalization import BatchNormalization
# from tensorflow.contrib.keras.python.keras.backend import learning_phase
from tensorflow.python.keras.backend import learning_phase

from nibabel import load as load_nii
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


# from npu_bridge.estimator import npu_ops

# SAVE_PATH = 'unet3d_baseline.hdf5'
# OFFSET_W = 16
# OFFSET_H = 16
# OFFSET_C = 4
# HSIZE = 64
# WSIZE = 64
# CSIZE = 16
# batches_h, batches_w, batches_c = (224-HSIZE)/OFFSET_H+1, (224-WSIZE)/OFFSET_W+1, (152 - CSIZE)/OFFSET_C+1


def parse_inputs():
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-r', '--root-path', dest='root_path', default='../ori_images/BRATS2017/Brats17ValidationData/')
    parser.add_argument('-m', '--model-path', dest='model_path',
                        default='NoneDense-0')
    parser.add_argument('-ow', '--offset-width', dest='offset_w', type=int, default=12)
    parser.add_argument('-oh', '--offset-height', dest='offset_h', type=int, default=12)
    parser.add_argument('-oc', '--offset-channel', dest='offset_c', nargs='+', type=int, default=12)
    parser.add_argument('-ws', '--width-size', dest='wsize', type=int, default=38)
    parser.add_argument('-hs', '--height-size', dest='hsize', type=int, default=38)
    parser.add_argument('-cs', '--channel-size', dest='csize', type=int, default=38)
    parser.add_argument('-ps', '--pred-size', dest='psize', type=int, default=12)
    parser.add_argument('-gpu', '--gpu', dest='gpu', type=str, default='0')
    parser.add_argument('-mn', '--model_name', dest='model_name', type=str, default='dense24')
    parser.add_argument('-nc', '--correction', dest='correction', type=bool, default=True)
    parser.add_argument('-input1', '--input_flair', dest='input_flair', default='../datasets/input_flair/')
    parser.add_argument('-input2', '--input_t1', dest='input_t1', default='../datasets/input_t1/')

    return vars(parser.parse_args())


options = parse_inputs()


# os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu']

def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def vox_generator_test(all_files):
    path = options['root_path']

    while 1:
        for file in all_files:
            print("==========file:", file)
            p = file
            if options['correction']:
                flair = load_nii(os.path.join(path, file, file + '_flair_corrected.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2_corrected.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1_corrected.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce_corrected.nii.gz')).get_data()
            else:
                flair = load_nii(os.path.join(path, p, p + '_flair.nii.gz')).get_data()

                t2 = load_nii(os.path.join(path, p, p + '_t2.nii.gz')).get_data()

                t1 = load_nii(os.path.join(path, p, p + '_t1.nii.gz')).get_data()

                t1ce = load_nii(os.path.join(path, p, p + '_t1ce.nii.gz')).get_data()
            data = np.array([flair, t2, t1, t1ce])
            data = np.transpose(data, axes=[1, 2, 3, 0])

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])

            labels = load_nii(os.path.join(path, p, p + '_seg.nii.gz')).get_data()

            yield data, data_norm, labels


def main():
    bin_path_1 = "../datasets/input_flair/"
    bin_path_2 = "../datssets/input_t1/"
    if not os.path.exists(bin_path_1):
        os.makedirs(bin_path_1)
    if not os.path.exists(bin_path_2):
        os.makedirs(bin_path_2)

    test_files = []
    DATA_PATH = options['root_path']
    with open(DATA_PATH + 'val.txt') as f:
        for line in f:
            test_files.append(line[:-1])

    num_labels = 5
    OFFSET_H = options['offset_h']
    OFFSET_W = options['offset_w']
    OFFSET_C = options['offset_c']
    HSIZE = options['hsize']
    WSIZE = options['wsize']
    CSIZE = options['csize']
    PSIZE = options['psize']
    SAVE_PATH = options['model_path']
    model_name = options['model_name']

    OFFSET_PH = (HSIZE - PSIZE) / 2
    OFFSET_PW = (WSIZE - PSIZE) / 2
    OFFSET_PC = (CSIZE - PSIZE) / 2

    batches_w = int(np.ceil((240 - WSIZE) / float(OFFSET_W))) + 1
    batches_h = int(np.ceil((240 - HSIZE) / float(OFFSET_H))) + 1
    batches_c = int(np.ceil((155 - CSIZE) / float(OFFSET_C))) + 1

    flair_t2_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))
    t1_t1ce_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))

    if model_name == 'dense48':

        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=t1_t1ce_node, name='t1')
    elif model_name == 'no_dense':

        flair_t2_15, flair_t2_27 = tf_models.PlainCounterpart(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.PlainCounterpart(input=t1_t1ce_node, name='t1')

    elif model_name == 'dense24':

        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat(input=t1_t1ce_node, name='t1')

    elif model_name == 'dense24_nocorrection':

        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat(input=t1_t1ce_node, name='t1')

    else:
        print(' No such model name ')

    t1_t1ce_15 = concatenate([t1_t1ce_15, flair_t2_15])
    t1_t1ce_27 = concatenate([t1_t1ce_27, flair_t2_27])

    t1_t1ce_15 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_15_cls')(t1_t1ce_15)
    t1_t1ce_27 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_27_cls')(t1_t1ce_27)

    t1_t1ce_score = t1_t1ce_15[:, 13:25, 13:25, 13:25, :] + \
                    t1_t1ce_27[:, 13:25, 13:25, 13:25, :]

    saver = tf.train.Saver()
    data_gen_test = vox_generator_test(test_files)
    dice_whole, dice_core, dice_et = [], [], []

    # for npu
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    # for npu

    with tf.Session(config=config) as sess:
        saver.restore(sess, SAVE_PATH)
        for i in range(len(test_files)):
            print('predicting %s' % test_files[i])
            # x, x_n, y = data_gen_test.next()
            x, x_n, y = next(data_gen_test)
            pred = np.zeros([240, 240, 155, 5])
            for hi in range(batches_h):
                offset_h = min(OFFSET_H * hi, 240 - HSIZE)
                # offset_ph = offset_h + OFFSET_PH
                offset_ph = int(offset_h + OFFSET_PH)
                for wi in range(batches_w):
                    offset_w = min(OFFSET_W * wi, 240 - WSIZE)
                    # offset_pw = offset_w + OFFSET_PW
                    offset_pw = int(offset_w + OFFSET_PW)
                    for ci in range(batches_c):
                        offset_c = min(OFFSET_C * ci, 155 - CSIZE)
                        # offset_pc = offset_c + OFFSET_PC
                        offset_pc = int(offset_c + OFFSET_PC)
                        data = x[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                        data_norm = x_n[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE,
                                    :]
                        data_norm = np.expand_dims(data_norm, 0)
                        if not np.max(data) == 0 and np.min(data) == 0:
                            score = sess.run(fetches=t1_t1ce_score,
                                             feed_dict={flair_t2_node: data_norm[:, :, :, :, :2],
                                                        t1_t1ce_node: data_norm[:, :, :, :, 2:],
                                                        learning_phase(): 0}
                                             )
                            flair = data_norm[:, :, :, :, :2]
                            t1 = data_norm[:, :, :, :, 2:]
                            flair1 = np.asarray(flair, dtype='float32')
                            t11 = np.asarray(t1, dtype='float32')
                            flair1.tofile(options['input_flair'] + "input_flair_" + str(i) + ".bin")
                            t11.tofile(options['input_t1'] + "input_t1_" + str(i) + ".bin")


if __name__ == '__main__':
    options = parse_inputs()
    # os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu']
    main()
