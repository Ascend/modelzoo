#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
import os
import six

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.inference as inference


## import our defined settings for bi-gru on NPU
from thumt.parameter_config import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--distribute", action="store_true",
                        help="Enable distributed training")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)

def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", ""],
        output="",
        record="",
        model="transformer",
        vocab=["", ""],

        # Default training hyper parameters
        batch_size=EVAL_BATCH_SIZE,
        train_decode_length=EVAL_DECODE_LENGTH,
        train_encode_length=EVAL_ENCODE_LENGTH,
        max_length=max(EVAL_DECODE_LENGTH, EVAL_ENCODE_LENGTH),
        num_threads=6,
        length_multiplier=1,
        mantissa_bits=2,
        warmup_steps=4000,
        train_steps=500000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        update_cycle=1,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        loss_scale=128,
        scale_l1=0.0,
        scale_l2=0.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=EVAL_BATCH_SIZE,
        eval_decode_length=EVAL_DECODE_LENGTH,
        eval_encode_length=EVAL_ENCODE_LENGTH,
        reference_num=REFERENCE_NUM,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        validation="",
        references=[""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False
    )

    return params

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        json_str = fd.readline()
        params.parse_json(json_str)

    return params

def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())

def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected

def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params

def override_parameters(params, args):
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.record = args.record or params.record
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocab[0]),
        "target": vocabulary.load_vocabulary(params.vocab[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }
    
    return params

def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]
    # print(inputs)
    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx
                if not isinstance(sym, six.string_types):
                    sym = sym.decode("utf-8")

            if sym == params.eos:
                break

            if sym == params.pad:
                break
            syms.append(sym)
        decoded.append(syms)

    return decoded

def restore_variables(checkpoint):
    # Load checkpoints
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}
    
    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)    
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]
        if name in values:
            ops.append(tf.assign(var, values[name]))
        else:
            print("Not in the Restore list: %s" % var.name)
    return tf.group(*ops, name="restore_op")


def main(args):
    if args.distribute:
        distribute.enable_distributed_training()

    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    # Validation
    if params.validation and params.references[0]:
        files = [params.validation] + list(params.references)
        eval_inputs = dataset.sort_and_zip_files(files)
        eval_input_fn = dataset.get_evaluation_input
    else:
        raise ValueError('You need determine the eval_inputs function')


    model = model_cls(params)
    


    def ckpt2pb():
        ## 1. transform ckpt to pt
        with tf.Graph().as_default():
            placeholders = {
                "source": tf.placeholder(tf.int32, [1, EVAL_ENCODE_LENGTH], "source"),
                "source_length": tf.placeholder(tf.int32, [1], "source_length")
            }
            eval_fn = model.get_inference_func(msame=True)
            predictions = tf.argmax(eval_fn(placeholders, params), axis=-1, name='translation')
            print(predictions.name)

            ckpt_path = "/data/tianda_BiGRU_tf_Wonderseen_with_sos_gru_refiner/bi-gru/train/eval/model.ckpt-545000"
            
            with tf.Session() as sess:
                tf.train.write_graph(sess.graph_def, './msame/pb_model', 'model.pb')    # 默认，不需要修改
                freeze_graph.freeze_graph(
                        input_graph='./msame/pb_model/model.pb',                        # 默认，不需要修改
                        input_saver='',
                        input_binary=False, 
                        input_checkpoint=ckpt_path, 
                        output_node_names="translation",#,rnnsearch/decoder/logit_sequence", # logit_sequence tensor is too large, not necessary
                        restore_op_name='save/restore_all',
                        filename_tensor_name='save/Const:0',
                        output_graph='./msame/pb_model/bigru.pb',                       # 改为对应网络的名称
                        clear_devices=False,
                        initializer_nodes='')
        

        ## 2. save input and output
        need_output = False ## sometimes not necessary to output the model output
        with tf.Graph().as_default():

            ckpt_path = "/data/tianda_BiGRU_tf_Wonderseen_with_sos_gru_refiner/bi-gru/train/"

            ## data pipeline
            input_fn = lambda: eval_input_fn(eval_inputs, params)
            features = input_fn()
            # features:
            # '''
            # {
            # 'source': <tf.Tensor 'IteratorGetNext:1' shape=(EVAL_BATCH_SIZE, EVAL_DECODE_LENGTH) dtype=int32>,
            # 'references': <tf.Tensor 'IteratorGetNext:0' shape=(EVAL_BATCH_SIZE, REFERENCE_NUM, EVAL_DECODE_LENGTH) dtype=string>,
            # 'source_length': <tf.Tensor 'IteratorGetNext:2' shape=(EVAL_BATCH_SIZE,) dtype=int32>
            # }
            # '''

            if need_output:
                restore_op = restore_variables(ckpt_path)
                eval_fn = model.get_inference_func()
                predictions = tf.argmax(eval_fn(features, params), axis=-1, name='translation')

            with tf.Session() as sess:
                if need_output:
                    sess.run(tf.global_variables_initializer())
                    sess.run(restore_op)
                train_step = 0
                try:
                    while True:
                        if need_output:
                            p, f = sess.run([predictions, features])
                        else:
                            f = sess.run(features)

                        for idx in range(EVAL_BATCH_SIZE):
                            if need_output:
                                p[idx:(idx+1), :].tofile("./msame/output_npu/{0:05d}.bin".format(train_step*EVAL_BATCH_SIZE+idx)) 
                            f["source"][idx:(idx+1), :].tofile("./msame/input/source/{0:05d}.bin".format(train_step*EVAL_BATCH_SIZE+idx))
                            f["source_length"][idx:(idx+1)].tofile("./msame/input/source_length/{0:05d}.bin".format(train_step*EVAL_BATCH_SIZE+idx))
                            np.savetxt("./msame/golden/references/{0:05d}.txt".format(train_step*EVAL_BATCH_SIZE+idx), [[str(tx, 'utf-8') for tx in f["references"][idx][0]]], fmt="%s")
                        train_step += 1
                except tf.errors.OutOfRangeError:
                    pass

    ckpt2pb()
    
    print("== done ==")

        
   
if __name__ == "__main__":
    main(parse_args())
