
from __future__ import print_function
from npu_bridge.npu_init import *
from data_reader import DataReader
from params import hparams
import tensorflow as tf
import time
import argparse
import os
import sys
import numpy as np
from scipy.io import wavfile
from datetime import datetime
from glow import WaveGlow, compute_waveglow_loss
from tensorflow.python.client import timeline
from data_reader import read_binary_lc

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt
STARTED_DATESTRING = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())

def get_arguments():

    def _str_to_bool(s):
        'Convert string to bool (in argparse context).'
        if (s.lower() not in ['true', 'false']):
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description='Parallel WaveNet Network')
    parser.add_argument('--filelist', type=str, default=None, required=True, help='filelist path for training data.')
    parser.add_argument('--wave_dir', type=str, default=None, required=True, help='wave data directory for training data.')
    parser.add_argument('--lc_dir', type=str, default=None, required=True, help='local condition directory for training data.')
    parser.add_argument('--ngpu', type=int, default=1, help='gpu numbers')
    parser.add_argument('--run_name', type=str, default='waveglow', help='run name for log saving')
    parser.add_argument('--restore_from', type=str, default=None, help='restore model from checkpoint')
    parser.add_argument('--store_metadata', type=_str_to_bool, default=False, help='Whether to store advanced debugging information')
    return parser.parse_args()

def write_wav(waveform, sample_rate, filename):
    '\n\n    :param waveform: [-1,1]\n    :param sample_rate:\n    :param filename:\n    :return:\n    '
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))

def save(saver, sess, logdir, step, write_meta_graph=True):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end='')
    sys.stdout.flush()
    if (not os.path.exists(logdir)):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=write_meta_graph)
    print(' Done.')

def load(saver, sess, logdir):
    print('Trying to restore saved checkpoints from {} ...'.format(logdir), end='')
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[(- 1)].split('-')[(- 1)])
        print('  Global step was: {}'.format(global_step))
        print('  Restoring...', end='')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(' Done.')
        return global_step
    else:
        print(' No checkpoint found.')
        return None

def average_gradients(tower_grads):
    '\n    Calculate the average gradient for each shared variable across all towers.\n    Note that this function provides a synchronization point across all towers.\n    Args:\n        tower_grads: List of lists of (gradient, variable) tuples. The outer list\n            is over individual gradients. The inner list is over the gradient\n            calculation for each tower.\n    Returns:\n        List of pairs of (gradient, variable) where the gradient has been averaged\n            across all towers.\n    '
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for (g, _) in grad_and_vars:
            if (g is None):
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        if (len(grads) == 0):
            average_grads.append((None, grad_and_vars[0][1]))
            continue
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def stats_graph(graph):
    flops = None
    params = None
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.tra))

def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def main():
    args = get_arguments()
    args.logdir = os.path.join(hparams.logdir_root, args.run_name)
    if (not os.path.exists(args.logdir)):
        os.makedirs(args.logdir)
    args.gen_wave_dir = os.path.join(args.logdir, 'wave')
    os.makedirs(args.gen_wave_dir, exist_ok=True)
    assert (hparams.upsampling_rate == hparams.hop_length), 'upsamling rate should be same as hop_length'
    coord = tf.train.Coordinator()
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(hparams.lr, global_step, hparams.decay_steps, 0.95, staircase=True)
    optimizer = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=learning_rate))
    with tf.device('/cpu:0'):
        with tf.name_scope('inputs'):
            reader = DataReader(coord, args.filelist, args.wave_dir, args.lc_dir)
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    # 手工迁移
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关

    sess = tf.Session(config=config)
    ###
    reader.start_threads()
    audio_placeholder = tf.placeholder(tf.float32, shape=[None, None, 1], name='audio')
    lc_placeholder = tf.placeholder(tf.float32, shape=[None, None, hparams.num_mels], name='lc')
    tower_losses = []
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(args.ngpu):
            with tf.device('/cpu:0'), tf.name_scope(('tower_%d' % i)):
                glow = WaveGlow(lc_dim=hparams.num_mels, n_flows=hparams.n_flows, n_group=hparams.n_group, n_early_every=hparams.n_early_every, n_early_size=hparams.n_early_size)
                print(('create network %i' % i))
                local_audio_placeholder = audio_placeholder[(i * hparams.batch_size):((i + 1) * hparams.batch_size), :, :]
                local_lc_placeholder = lc_placeholder[(i * hparams.batch_size):((i + 1) * hparams.batch_size), :, :]
                (output_audio, log_s_list, log_det_W_list) = glow.create_forward_network(local_audio_placeholder, local_lc_placeholder)
                loss = compute_waveglow_loss(output_audio, log_s_list, log_det_W_list, sigma=hparams.sigma)
                grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
                tower_losses.append(loss)
                tower_grads.append(grads)
                tf.summary.scalar(('loss_tower_%d' % i), loss)
    print('create network finished')
    loss = tf.reduce_mean(tower_losses)
    averaged_gradients = average_gradients(tower_grads)
    train_ops = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(args.logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        lc_placeholder_infer = tf.placeholder(tf.float32, shape=[1, None, hparams.num_mels], name='lc_infer')
        audio_infer_ops = glow.infer(lc_placeholder_infer, sigma=hparams.sigma)
    init = tf.global_variables_initializer()
    sess.run(init)
    print('parameters initialization finished')
    total_parameters = count()
    print('######################################################')
    print('### Total Trainable Params is {} ###'.format(total_parameters))
    print('######################################################')
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=30)
    saved_global_step = 0
    if (args.restore_from is not None):
        try:
            saved_global_step = load(saver, sess, args.restore_from)
            if (saved_global_step is None):
                saved_global_step = 0
        except Exception:
            print('Something went wrong while restoring checkpoint. We will terminate training to avoid accidentally overwriting the previous model.')
            raise
        print('restore model successfully!')
    print('start training.')
    last_saved_step = saved_global_step
    try:
        for step in range((saved_global_step + 1), hparams.train_steps):
            (audio, lc) = reader.dequeue(num_elements=(hparams.batch_size * args.ngpu))
            if (hparams.lc_conv1d or hparams.lc_encode or hparams.transposed_upsampling):
                lc = np.reshape(lc, [(hparams.batch_size * args.ngpu), (- 1), hparams.num_mels])
            else:
                lc = np.tile(lc, [1, 1, hparams.upsampling_rate])
                lc = np.reshape(lc, [(hparams.batch_size * args.ngpu), (- 1), hparams.num_mels])
            start_time = time.time()
            if (((step % 100) == 0) and args.store_metadata):
                print('Storing metadata')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                (summary, loss_value, _, lr) = sess.run([summaries, loss, train_ops, learning_rate], feed_dict={audio_placeholder: audio, lc_placeholder: lc}, options=run_options, run_metadata=run_metadata)
                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(args.logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                (summary, loss_value, _, lr) = sess.run([summaries, loss, train_ops, learning_rate], feed_dict={audio_placeholder: audio, lc_placeholder: lc})
                writer.add_summary(summary, step)
            duration = (time.time() - start_time)
            step_log = 'step {:d} - loss = {:.3f}, lr={:.8f}, time cost={:4f}'.format(step, loss_value, lr, duration)
            print(step_log)
            if ((step % hparams.save_model_every) == 0):
                save(saver, sess, args.logdir, step)
                last_saved_step = step
            if ((step % hparams.gen_test_wave_every) == 0):
                generate_wave(lc_placeholder_infer, audio_infer_ops, sess, step, args.gen_wave_dir)
    except KeyboardInterrupt:
        print()
    finally:
        if (step > last_saved_step):
            save(saver, sess, args.logdir, step)
        coord.request_stop()
        coord.join()

def generate_wave(lc_placeholder_infer, audio_infer_ops, sess, step, path):
    save_name = (str(step).zfill(8) + '.wav')
    save_name = os.path.join(path, save_name)
    lc = read_binary_lc(hparams.gen_file, hparams.num_mels)
    if (hparams.lc_conv1d or hparams.lc_encode or hparams.transposed_upsampling):
        lc = np.reshape(lc, [1, (- 1), hparams.num_mels])
    else:
        lc = np.tile(lc, [1, 1, hparams.upsampling_rate])
        lc = np.reshape(lc, [1, (- 1), hparams.num_mels])
    audio_output = sess.run(audio_infer_ops, feed_dict={lc_placeholder_infer: lc})
    audio_output = audio_output.flatten()
    write_wav(audio_output, hparams.sample_rate, save_name)
if (__name__ == '__main__'):
    main()
