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

import argparse
import datetime
import glob
import math
import npu_bridge
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import low_high_model

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--print_interval', type=int, default=300, help='The number of steps between each validation')
parser.add_argument('--n_generator', type=int, default=5, help='Update ratio between discriminator and generator')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight of mse loss')
parser.add_argument('--beta', type=float, default=0.05, help='Weight of generator loss')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing the dataset')
parser.add_argument('--model_dir', type=str, default='./output', help='Directory for storing the training output')
args = parser.parse_args()


def get_data(npy_path):
    """Load data from the given path

    The data is saved in an npy file, which should be loaded by `np.load` function.
    The loaded data is stored as Python's `dict` type, where the key 'sample' corresponds to low-resolution images,
    and the key 'label' corresponds to high-resolution images.

    Args:
        npy_path: The path of npy data file.

    Returns:
        low-resolution and high-resolution data, both of which are of type ndarray.
        sample_array.shape: [data_size, 16, 16, 3]
        label_array.shape: [data_size, 64, 64, 3]

    """
    data = np.load(npy_path, allow_pickle=True)
    data_dict = data.item()
    sample_imgs = data_dict['sample']
    label_imgs = data_dict['label']

    sample_array = np.array(sample_imgs, dtype=np.float32)
    label_array = np.array(label_imgs, dtype=np.float32)

    sample_array = np.clip(sample_array / 255., 0., 1.)
    label_array = np.clip(label_array / 255., 0., 1.)

    return sample_array, label_array


def get_next(batch_id, batch_size, sample_array, label_array):
    """Get a batch of data

    Args:
        batch_id: The number of the current batch.
        batch_size: The size of batch data.
        sample_array : low-resolution data of shape [data_size, 16, 16, 3]
        label_array: high-resolution data of shape [data_size, 64, 64, 3]

    Returns:
        Batch data and next batch's id.

    """
    last_index = batch_id + batch_size
    batch_sample = sample_array[batch_id: last_index]
    batch_label = label_array[batch_id: last_index]
    return batch_sample, batch_label, last_index


def evaluate(sess, eval_model, npy_paths):
    """Model evaluation

    Args:
        sess: Tensorflow Session.
        eval_model: Model instance for evaluation.
        npy_paths: Path for evaluation dataset.

    Returns:
        PSNR on the evaluation dataset

    """
    sum_psnr, cnt = 0, 0
    for npy, npy_path in enumerate(npy_paths):
        sample_array, label_array = get_data(npy_path)
        batch_num = math.floor(sample_array.shape[0] / args.batch_size)
        batch_id = 0
        for step in range(1, int(batch_num) + 1):
            lr_img, hr_img, batch_id = get_next(batch_id, args.batch_size, sample_array, label_array)
            feed_dict = {eval_model.hr_img: hr_img, eval_model.lr_img: lr_img, eval_model.is_train: False}
            batch_psnr = sess.run(eval_model.PSNR, feed_dict)
            sum_psnr += batch_psnr
            cnt += 1
    avg_psnr = sum_psnr / cnt
    return round(avg_psnr, 4)


def train():
    """Model training

    Train model on the training dataset. After training, the model with the best
    performance on the development dataset will be saved in the directory specified by `args.model_dir`.

    """
    # Loading training dataset and development dataset
    # The dataset is stored in several npy files, each of which contains part of the dataset.
    train_npy_paths = glob.glob(os.path.join(args.data_dir, 'train', '*.npy'))
    dev_npy_paths = glob.glob(os.path.join(args.data_dir, 'dev', '*.npy'))
    train_npy_paths.sort()
    dev_npy_paths.sort()

    global_steps = 0
    model = low_high_model.MODEL(args)
    model.build_optimizer()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Config for Ascend 910
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    tf.logging.set_verbosity(tf.logging.INFO)
    n_finetune = 0
    max_psnr = 0

    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    for epo in range(args.max_epoch + 10):  # The model is fine-tuned for another 2000 generator updates in the end.
        for npy, npy_path in enumerate(train_npy_paths):
            sample_array, label_array = get_data(npy_path)
            batch_num = math.floor(sample_array.shape[0] / args.batch_size)
            batch_id = 0
            for step in range(1, int(batch_num) + 1):
                lr_img, hr_img, batch_id = get_next(batch_id, args.batch_size, sample_array, label_array)
                feed_dict = {model.hr_img: hr_img, model.lr_img: lr_img, model.is_train: True}
                _ = sess.run(model.dis_train_op, feed_dict)
                if global_steps % args.n_generator == 0 or epo > args.max_epoch:
                    _ = sess.run(model.gen_train_op, feed_dict)
                    if epo > args.max_epoch:
                        n_finetune += 1
                if n_finetune > 2000:
                    return

                global_steps += 1

                # Every `args.print_interval` training steps, the model will be evaluated on the development dataset.
                # The model with the best performance will be saved.
                if global_steps % args.print_interval == 0:
                    (generator_cost, mse_cost, discrim_cost) = sess.run(
                        [model.generator_cost,
                         model.mse_loss,
                         model.discrim_cost], feed_dict)

                    psnr = evaluate(sess, model, dev_npy_paths)
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('{} epo={},npy={},gen_loss={:.4f},dis_loss={:.5f},mse_loss={:.3f},step={},psnr={}'.format(
                        current_time, epo, npy, generator_cost, discrim_cost, mse_cost, global_steps, psnr))
                    with open(os.path.join(args.model_dir, 'loss_acc.txt'), 'a') as loss_acc_file:
                        loss_acc_file.write('step={} psnr={} gen_loss={:.4f} dis_loss={:.5f} mse_loss={:.3f}\n'.format(
                            global_steps, psnr, generator_cost, discrim_cost, mse_cost))

                    if max_psnr < psnr:
                        max_psnr = psnr
                        best_model_path = os.path.join(args.model_dir, 'model')
                        tf.train.Saver(var_list=var_list).save(sess, best_model_path, global_step=0)
                        print(psnr, '------------------ best model found ----------------')


def test():
    """Model testing

    Test the model on the test dataset.
    The model should be saved in the directory specified by `args.model_dir`

    """
    # Loading test dataset
    test_npy_paths = glob.glob(os.path.join(args.data_dir, 'test', '*.npy'))
    test_npy_paths.sort()

    model = low_high_model.MODEL(args)
    model.build_optimizer()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Config for Ascend 910
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    tf.logging.set_verbosity(tf.logging.INFO)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    save_path = os.path.join(args.model_dir, os.path.basename(ckpt.model_checkpoint_path))
    saver.restore(sess, save_path)
    psnr = evaluate(sess, model, test_npy_paths)
    print('test psnr:', psnr)


if __name__ == '__main__':
    # Fix random seed to get stable results
    random_seed = 0
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # Set TensorFlow's log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if args.train:
        train()
    if args.test:
        test()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
