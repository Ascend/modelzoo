# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import os
from glob import glob
from networks import encoder, decoder, converter
from utils import *
from modules import *
import hparams
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse
import pickle


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train')
    arg_parser.add_argument('--data_path', type=str)
    arg_parser.add_argument('--local_path', type=str)
    arg_parser.add_argument('--checkpoint', type=str, help='prefix of saved checkpoint')
    arg_parser.add_argument('--max_text_len', type=int, default=188)
    arg_parser.add_argument('--max_mel_len', type=int, default=870)
    arg_parser.add_argument('--vocab_size', type=int, default=148)
    args = arg_parser.parse_args()

    hyper_params = hparams.create_hparams()
    vocab_len = args.vocab_size
    batch_size = hyper_params.batch_size

    text_seq = tf.placeholder(shape=[1, 188], dtype=tf.int32)
    mel_target = tf.zeros(shape=[1, 870, 80], dtype=tf.float32)

    sample_list = [os.path.join(args.data_path, fi) for fi in os.listdir(args.data_path) if fi.endswith('.pkl')]

    # prepare placeholder
    encoded = encoder(text_seq, training=False)
    mel_output, bef_mel_output, done_output, decoder_state, LTSM, step \
        = decoder(mel_target, encoded, training=False)

    # build session
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # sess.run(tf.initializers.global_variables())
        saver.restore(sess=sess, save_path=os.path.join(args.local_path, args.checkpoint))
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=['decoder/add', 'decoder/done_output/Sigmoid']
        )
        tf.io.write_graph(
            output_graph_def,
            logdir='./convert_log',
            name='./tacotron2.pb',
            as_text=False
        )
        count = 0
        for si in sample_list:
            with open(si, 'rb') as fr:
                sp = pickle.load(fr)
                out_mel_pst, gate_output = sess.run([mel_output, done_output], feed_dict={
                    text_seq: np.expand_dims(sp['padded_text'], axis=0)
                })
                print('sample: ', si)
                print('mel_output: ', out_mel_pst)
                print('gate_output: ', gate_output)
                count += 1
                if count == 10:
                    break
