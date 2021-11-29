# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#! -*- encoding: utf-8 -*-

from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

# from data_reader import DataReader
from params import hparams
import glob

import time
import argparse
import os
import sys

from scipy.io import wavfile
from datetime import datetime
from glow import WaveGlow, compute_waveglow_loss
from tensorflow.python.client import timeline
from ljspeech_to_tfrecords import get_tfrecords_dataset
import librosa
from audio_utils import melspectrogram


from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu import npu_scope
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.hccl import hccl_ops



STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveGlow Network')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                        help='init learning rate value')
    parser.add_argument('--epochs', type=int, default=110,
                        help='epochs value( use repeat sample 10x data, similar run 1000 epochs)')
    parser.add_argument('--decay_steps', type=int, default=2000,
                        help='learning rate decay steps value')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size value')

    parser.add_argument('--ngpu', type=int, default=1, help='gpu numbers')
    parser.add_argument('--run_name', type=str, default='waveglow',
                        help='run name for log saving')
    parser.add_argument('--infer_raw_audio_filepath', type=str, default='./LJSpeech-1.1/wavs/LJ001-0001.wav',
                        help='infer_raw_audio_filepath')
    #"./logdir/waveglow/model.ckpt-160000"
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    parser.add_argument('--store_metadata', type=_str_to_bool, default=False,
                        help='Whether to store advanced debugging information')


    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    """

    :param waveform: [-1,1]
    :param sample_rate:
    :param filename:
    :return:
    """
    # TODO: write wave to 16bit PCM, don't use librosa to write wave
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


def save(saver, sess, logdir, step, write_meta_graph=True):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=write_meta_graph)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")
    print("  Checkpoint found: {}".format(logdir))
    global_step = int(logdir
                      .split('/')[-1]
                      .split('-')[-1])
    print("  Global step was: {}".format(global_step))
    print("  Restoring...", end="")
    saver.restore(sess, logdir)
    print(" Done.")
    return global_step

    """
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None
    """

def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1)... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if len(grads) == 0:
            average_grads.append((None, grad_and_vars[0][1]))
            continue

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.tra))


def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def main():

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("tf version:",tf.__version__)
    args = get_arguments()
    hparams.lr=args.learning_rate
    hparams.epochs=args.epochs
    hparams.decay_steps = args.decay_steps
    hparams.batch_size = args.batch_size

    rank_size = int(os.environ.get('RANK_SIZE', '').strip())
    args.ngpu=int(os.getenv('RANK_SIZE'))
    deviceid=int(os.getenv('DEVICE_ID'))

    print("#########gpu number:",args.ngpu)
    args.logdir = os.path.join(hparams.logdir_root, args.run_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    args.gen_wave_dir = os.path.join(args.logdir, 'wave')
    os.makedirs(args.gen_wave_dir, exist_ok=True)

    assert hparams.upsampling_rate == hparams.hop_length, 'upsamling rate should be same as hop_length'

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    #custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["mix_compile_mode"].b = True
    # Autotune
    custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes(os.getenv("FLAG_AUTOTUNE"))

    custom_op.parameter_map["hcom_parallel"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关

    sess = tf.Session(config=config)

    global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0),dtype=tf.int32, trainable=False)

    #暂时不用学习率衰减
    learning_rate = tf.train.exponential_decay(hparams.lr, global_step, hparams.decay_steps, 0.93, staircase=True)
    learning_rate=tf.maximum(learning_rate,1.0e-4)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,
                                                               decr_every_n_nan_or_inf=2, decr_ratio=0.5)

    #loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.bert_loss_scale)

    # device数是否大于1，如果大于1，进行分布式训练
    optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager,
                                      is_distributed=True if rank_size > 1 else False)

    train_filelist = glob.glob(os.path.join(hparams.tfrecords_dir, "ljs_train*.tfrecords"))
    train_input_fn = get_tfrecords_dataset(train_filelist, hparams, num_cpu_threads=3,
                                           batch_size=hparams.batch_size , is_training=True,
                                           drop_remainder=True)
    traindata_iterator = train_input_fn().make_initializable_iterator()
    sess.run(traindata_iterator.initializer)
    mel_spec=prepare_one_infer_data(args.infer_raw_audio_filepath, hparams)

    glow = WaveGlow(lc_dim=hparams.num_mels,
            n_flows=hparams.n_flows,
            n_group=hparams.n_group,
            n_early_every=hparams.n_early_every,
            n_early_size=hparams.n_early_size,
            traindata_iterator=traindata_iterator,
            testdata_iterator=None # testdata_iterator
            )
    print('create network...')

    output_audio, log_s_list, log_det_W_list,batch_data = glow.create_forward_network()
    loss = compute_waveglow_loss(output_audio, log_s_list, log_det_W_list, sigma=hparams.sigma)

    total_loss=loss
    if rank_size>1:
        total_loss = tf.div(hccl_ops.allreduce(loss, "sum"),int(rank_size))

    audio_batch = batch_data["wav"]
    lc_batch = batch_data["mel"]

    # inference for audio
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        lc_placeholder_infer = tf.placeholder(tf.float32, shape=[1, mel_spec.shape[1], hparams.num_mels], name='lc_infer')
        audio_infer_ops = glow.infer(lc_placeholder_infer, sigma=hparams.sigma)

    print("create network finished")

    all_trainable_variables = tf.trainable_variables()
    grads_vars= optimizer.compute_gradients(loss,all_trainable_variables )
    grads = [grad for (grad, var) in grads_vars  if grad is not None ]
    params = [var for (grad, var) in grads_vars  if grad is not None]

    #mask_grads=[mask_nan(grad) for grad in grads]
    #grads=mask_grads

    grads, norm = tf.clip_by_global_norm(grads, 1.0)

    if int(rank_size) > 1:
        reduce_grads = [hccl_ops.allreduce(grad, "sum") for grad in grads]
        grads=[tf.div(grad, int(rank_size)) for grad in reduce_grads ]

    train_ops = optimizer.apply_gradients(list(zip(grads, params)), global_step=global_step)

    init = tf.global_variables_initializer()
    sess.run(init)
    print('parameters initialization finished')

    if int(rank_size) > 1:
        op_list=[]
        all_trainable_variables_broadcast = hccl_ops.broadcast(all_trainable_variables, 0)
        for vid,var in enumerate(all_trainable_variables_broadcast):
            op_list.append(tf.assign(all_trainable_variables[vid],var))
        sess.run(op_list)

    if deviceid==0:    
        tf.summary.scalar('total_loss', total_loss)

        # Set up logging for TensorBoard.
        writer = tf.summary.FileWriter(args.logdir)
        writer.add_graph(tf.get_default_graph())
        summaries = tf.summary.merge_all()

    total_parameters = count()
    print("######################################################")
    print("### Total Trainable Params is {} ###".format(total_parameters))
    print("######################################################")

    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=30)

    saved_global_step = 0
    if args.restore_from is not None:
        try:
            saved_global_step = load(saver, sess, args.restore_from)
            if saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = 0
        except Exception:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        print("restore model successfully!")

    print('start training.')
    steps_per_epoch=int(134380/(hparams.batch_size * args.ngpu))
    print("steps_per_epoch:",steps_per_epoch)
    last_saved_step = saved_global_step

    for step in range(saved_global_step + 1, hparams.train_steps):
        start_time = time.time()

        if deviceid==0:
            temp_summarys,loss_value, _, lr, gstep = sess.run([summaries,total_loss, train_ops, learning_rate, global_step])
            writer.add_summary(temp_summarys,gstep)
            duration = time.time() - start_time
            step_log = 'epoch {:d} - step {:d} - loss = {:.3f}, lr={:.8f}, time cost={:4f}, samples per second={:4f},global step:{:d}' \
                .format(int(step / steps_per_epoch) + 1, step, loss_value, lr, duration,
                        args.ngpu * hparams.batch_size / duration, gstep)
            print(step_log)

            if step % hparams.save_model_every == 0:
                save(saver, sess, args.logdir, step)
                last_saved_step = step
                print("inference generate one audio after train %d steps..."%(step))
                generate_wave(lc_placeholder_infer, audio_infer_ops, sess, step, args.logdir,mel_spec)
        else:
            loss_value, _, lr, gstep = sess.run([total_loss, train_ops, learning_rate, global_step])

        if int(step/steps_per_epoch)+1>args.epochs:
            print("training max epochs,now stop training...")
            # raise Exception("training finish...")
            break

    if step > last_saved_step:
        save(saver, sess, args.logdir, step)


def prepare_one_infer_data(infer_raw_audio_filepath,params):
    ###prepare infer data
    # read wave
    audio, sample_rate = librosa.load(infer_raw_audio_filepath, sr=None, mono=True)

    if sample_rate != params.sample_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sample_rate, params.sample_rate))
    # compute mel spectrogram
    mel_spec = melspectrogram(audio)

    mel_spec = np.array(mel_spec, 'float32')
    assert mel_spec.size % float(params.num_mels) == 0.0, \
        'specified dimension %s not compatible with data' % (params.num_mels,)
    mel_spec = mel_spec.reshape((1, -1, params.num_mels))

    #print(mel_spec)
    return mel_spec



def generate_wave(lc_placeholder_infer,audio_infer_ops, sess, step, path,mel_spec):
    save_name = str(step).zfill(8) + '.wav'
    save_name = os.path.join(path, save_name)

    audio_output = sess.run(audio_infer_ops, feed_dict={lc_placeholder_infer: mel_spec} )
    audio_output = audio_output[-1].flatten()
    write_wav(audio_output, hparams.sample_rate, save_name)

def mask_nan(x,value=0.0):
    '''
    用value值来代替nan 或inf
    '''
    x_values= tf.add(tf.zeros_like(x),value)
    mask = tf.math.is_finite(x)

    #mask1=tf.reduce_any(tf.abs(x)<1000.0)
    #total_mask=tf.logical_and(mask,mask1)
    y = tf.where(mask, x, x_values)

    return y




if __name__ == '__main__':
    main()
