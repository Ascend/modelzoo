from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


import numpy as np
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models

import numpy as np
import thumt.utils.bleu as bleu
import six

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
    parser.add_argument("--model", type=str, default="rnnsearch",
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

def main(args):
    ## params massive for reloading vocabulary
    model_cls = models.get_model(args.model)
    params = default_parameters()
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)


    ## Start computing BLEU...
    msame_output_dir = "msame/output_offline/"
    msame_golden_dir = "msame/golden/references/"
    all_outputs, all_refs = [], []
    total_sample_num = len([x for x in os.listdir(os.path.dirname(msame_golden_dir))])

    def read_txt(filename, dtype_func):
        with open(filename, 'r') as f:
            if dtype_func != str:
                return [[dtype_func(tx) for tx in d.split()] for d in f.readlines()]
            else:
                return [d.split() for d in f.readlines()]

    for idx in range(total_sample_num):
        msame_gold = read_txt(os.path.join(msame_golden_dir, "{0:05d}.txt".format(idx)), dtype_func=str)
        msame_translation = read_txt(os.path.join(msame_output_dir, "{0:05d}_output_0.txt".format(idx)), dtype_func=int)
        # msame_sequence_logit = read_txt_to_number(os.path.join(msame_output_dir, "{0:05d}_output_1.txt".format(idx)), dtype_func=float) # not necessary

        ##
        all_outputs.extend(msame_translation)
        all_refs.extend(msame_gold)

    ## Prediction
    decoded_symbols = decode_target_ids(all_outputs, params)
    for i, l in enumerate(decoded_symbols):
        decoded_symbols[i] = " ".join(l).replace("@@ ", "").split()

    ## References
    decoded_refs = [decode_target_ids(refs, params) for refs in [all_refs]]
    decoded_refs = [list(x) for x in zip(*decoded_refs)]


    print("############### MSAME BLEU testing #############")
    print("Golden Translation at:", msame_golden_dir)
    print("BiGRU Translation at:", msame_output_dir)
    print("BLEU =", bleu.bleu(decoded_symbols, decoded_refs)*100.)

if __name__ == "__main__":
    main(parse_args())
