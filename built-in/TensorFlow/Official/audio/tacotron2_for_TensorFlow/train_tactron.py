# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import os
from glob import glob
from networks import encoder, decoder, converter
# from utils import *
from modules import *
import hparams
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import sample_generator
import argparse
from npu_bridge.estimator import npu_ops
import time
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer


def input_fn_builder(folder, hyper_params):
    gen_fn = sample_generator.create_generator(data_path=folder)

    def input_fn():

        def parser(record1, record2, record3, record4):
            return {'text': record1, 'mask': record2}, {'mels': record3, 'gate': record4}

        dataset = tf.data.Dataset.from_generator(
            generator=gen_fn,
            output_types=(tf.float32, tf.bool, tf.float32, tf.int32),
            output_shapes=(
                tf.TensorShape([args.max_text_len, ]),
                tf.TensorShape([args.max_text_len, ]),
                tf.TensorShape([args.max_mel_len, hyper_params.n_mel_channels]),
                tf.TensorShape([args.max_mel_len, ])
            )
        )
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record1, record2, record3, record4: parser(record1, record2, record3, record4),
                batch_size=hyper_params.batch_size,
                num_parallel_batches=4,
                drop_remainder=True
            )
        )
        return dataset

    return input_fn


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train')
    arg_parser.add_argument('--data_path', type=str)
    arg_parser.add_argument('--steps', type=int)
    arg_parser.add_argument('--local_path', type=str)
    arg_parser.add_argument('--max_text_len', type=int, default=188)
    arg_parser.add_argument('--max_mel_len', type=int, default=870)
    arg_parser.add_argument('--vocab_size', type=int, default=148)
    arg_parser.add_argument('--save_checkpoints_steps', type=int, default=1000)
    arg_parser.add_argument('--log_every_n_step', type=int, default=10)
    arg_parser.add_argument('--start_decay', type=int, default=10000)
    arg_parser.add_argument('--decay_steps', type=int, default=10000)
    arg_parser.add_argument('--decay_rate', type=float, default=10000)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--multi_npu', type=int, default=0)
    arg_parser.add_argument('--rank_id', type=int)
    arg_parser.add_argument('--rank_size', type=int, default=1)
    arg_parser.add_argument('--warm_start', type=str)
    args = arg_parser.parse_args()

    if not os.path.exists(args.local_path):
        os.makedirs(args.local_path)

    hyper_params = hparams.create_hparams()
    vocab_len = args.vocab_size
    batch_size = hyper_params.batch_size

    in_fn = input_fn_builder(folder=args.data_path, hyper_params=hyper_params)
    ds = in_fn()
    if args.multi_npu == 1:
        ds = ds.shard(args.rank_size, args.rank_id)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=args.lr, shape=[], dtype=tf.float32)
    # need to build another learning rate decay schedule
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step - args.start_decay,  # lr = 1e-3 at step 50k
        args.decay_steps,
        args.decay_rate,  # lr = 1e-5 around step 310k
        name='lr_exponential_decay')

    itor = ds.make_one_shot_iterator()
    feature, label = itor.get_next()
    text_seq = feature['text']
    bs = int(text_seq.shape[0])
    print('input name: ', text_seq.name)
    mel_target = label['mels']
    gate = label['gate']
    encoded = encoder(text_seq, training=True)
    mel_output, bef_mel_output, done_output, decoder_state, LTSM, step \
        = decoder(mel_target, encoded, training=True)
    print('\noutput name: ', mel_output.name, '\n')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if args.multi_npu == 1:
        optimizer = NPUDistributedOptimizer(optimizer)
    mel_loss = tf.losses.mean_squared_error(labels=mel_target, predictions=mel_output) + \
        tf.losses.mean_squared_error(labels=mel_target, predictions=bef_mel_output)
    gate_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=done_output, labels=gate))
    train_loss = mel_loss + gate_loss
    # create train op
    grads_and_vars = optimizer.compute_gradients(train_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # train_op = optimizer.minimize(loss=train_loss, global_step=global_step)

    # build session
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for i in range(args.steps):
            start = time.time()
            _, loss_value = sess.run([train_op, train_loss])
            end = time.time()
            if i % args.log_every_n_step == 0:
                print('step time in second: %f' % (end - start))
                print('samples per second: %f' % (bs * args.rank_size / (end - start)))
                print('step: %d, loss: %f' % (i, float(loss_value)))
            if i % args.save_checkpoints_steps == 0:
                saver.save(sess, os.path.join(args.local_path, 'tacotron2'), global_step=i)
