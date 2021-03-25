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
import tensorflow.compat.v1 as tf1
import thumt.data.dataset as dataset
import thumt.data.record as record
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.distribute as distribute
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimizers as optimizers
import thumt.utils.parallel as parallel

## NPU optimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

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
    parser.add_argument("--half", action="store_true",
                        help="Enable FP16 training")
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
        num_threads=6,
        batch_size=EVAL_BATCH_SIZE,
        train_decode_length=TRAIN_DECODE_LENGTH,
        train_encode_length=TRAIN_ENCODE_LENGTH,
        max_length=max(TRAIN_DECODE_LENGTH, TRAIN_ENCODE_LENGTH),
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



def clip_gradients(grads, norm=None, mixed_precision=False):
    """Clips gradients by global norm."""
    # clipped_grads = [tf.clip_by_norm(grad, float(norm)) for grad in grads]
    # return clipped_grads   
    
    if not mixed_precision:
        clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
    else:
        all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
        # to prevent clip_by_global_norm from having a hizzy fit.

        clipped_grads, _ = tf.clip_by_global_norm(
            grads, norm,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)))
        
    return clipped_grads

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
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


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")



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
    if not checkpoint:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
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
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))
        else:
            print("Not in the Restore list: %s" % var.name)
    return ops
    # return tf.group(*ops, name="restore_op")


def print_variables():
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)


def main(args):
    if args.distribute:
        distribute.enable_distributed_training()

    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    if distribute.rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(
            params.output,
            "%s.json" % args.model,
            collect_params(params, model_cls.get_parameters())
        )

    # Build Graph
    with tf.Graph().as_default():
        if not params.record:
            # Build input queue
            features = dataset.get_training_input(params.input, params)
        else:
            features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )

        config = session_config_fn(params)

        ## Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        model = model_cls(params)

        ## Create global step
        global_step = tf.train.get_or_create_global_step()

        ## Multi-GPU setting
        sharded_losses = parallel.parallel_model(
            model.get_training_func(initializer, regularizer, dtype),
            features,
            params.device_list
        )
        test_feature = sharded_losses[0][1]
        if isinstance(sharded_losses[0], tuple):
            sharded_losses = [s[0] for s in sharded_losses]

        if distribute.rank() == 0:
            print_variables()
            if STOP_SEE_VARIABLE:
                input('======= input ==========')
        if TEST_INFERENCE:
            train_op = sharded_losses
            train_hooks = [tf.train.StopAtStepHook(last_step=params.train_steps)]
        else:
            loss = tf.add_n(sharded_losses) / len(sharded_losses)
            loss = loss + tf.losses.get_regularization_loss()
            learning_rate = get_learning_rate_decay(params.learning_rate, global_step, params)
            learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)

            # Create optimizer
            if params.optimizer == "Adam":
                opt = tf.train.AdamOptimizer(learning_rate,
                                             beta1=params.adam_beta1,
                                             beta2=params.adam_beta2,
                                             epsilon=params.adam_epsilon)
            elif params.optimizer == "LazyAdam":
                opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                       beta1=params.adam_beta1,
                                                       beta2=params.adam_beta2,
                                                       epsilon=params.adam_epsilon)
            else:
                raise RuntimeError("Optimizer %s not supported" % params.optimizer)

            ## GPU
            opt = optimizers.MultiStepOptimizer(opt, params.update_cycle)

            ## NPU distribute (not used when training bi-GRU)
            # if args.distribute:
            #     opt = NPUDistributedOptimizer(opt)

            ## loss scale
            if True:#args.half:
                ## GPU and NPU (static/fixed loss scale which is stable when training on NPU)
                opt = optimizers.LossScalingOptimizer(opt, params.loss_scale)

                ## NPU (original implement of static/fixed loss scale of Ascend910, which may be stable when training NPU, not yet test this version completely)
                # loss_scale_manager = FixedLossScaleManager(loss_scale=params.loss_scale)
                # loss_scale = loss_scale_manager.get_loss_scale()
                # tf.summary.scalar("loss_scale", loss_scale)
                # opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=args.distribute)

            # Optimization
            grads_and_vars = opt.compute_gradients(loss, colocate_gradients_with_ops=True)

            if params.clip_grad_norm:
                ## GPU adn NPU (*stable)
                grads, var_list = list(zip(*grads_and_vars)) 
                grads, _ = tf.clip_by_global_norm(grads, params.clip_grad_norm)
                grads_and_vars = zip(grads, var_list)

                ## NPU (not stable)
                # grads_and_vars = list(filter(
                #    lambda x:(
                #   x[0] is not None
                #   ),
                #   grads_and_vars
                # ))
                # grads, var_list = list(zip(*grads_and_vars))
                # grads = clip_gradients(grads, norm=params.clip_grad_norm, mixed_precision=True)
                # grads_and_vars = list(zip(grads, var_list))
                
            ## GPU
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

            ## NPU
            # train_op = opt.apply_gradients(grads_and_vars)
            # with tf.control_dependencies([train_op]):
            #    train_op = tf.assign_add(global_step, 1)

            tf.summary.scalar("loss", loss)
            tf.summary.scalar("learning_rate", learning_rate)

            # Hooks
            train_hooks = [
                tf.train.StopAtStepHook(last_step=params.train_steps),
                tf.train.NanTensorHook(loss),
                tf.train.LoggingTensorHook(
                    {
                        "step": global_step,
                        "loss": loss,
                        "learning_rate": learning_rate,
                        "source": tf.shape(features["source"]),
                        "target": tf.shape(features["target"])
                    },
                    every_n_iter=1
                )
            ]
                    

        # Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None
        if distribute.rank() == 0:
            # Add hooks
            save_vars = tf.trainable_variables() + [global_step]
            saver = tf.train.Saver(
                var_list=save_vars if params.only_save_trainable else None,
                max_to_keep=params.keep_checkpoint_max,
                sharded=False
            )
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            train_hooks.append(
                hooks.MultiStepHook(
                    tf.train.CheckpointSaverHook(
                        checkpoint_dir=params.output,
                        save_secs=params.save_checkpoint_secs or None,
                        save_steps=params.save_checkpoint_steps or None,
                        saver=saver),
                    step=params.update_cycle)
            )
    
            if False:#eval_input_fn is not None: ## NPU memory is not enough for training and evaluation at the same card
                train_hooks.append(
                    hooks.MultiStepHook(
                        hooks.EvaluationHook(
                            lambda f: inference.create_inference_graph(
                                [model], f, params
                            ),
                            lambda: eval_input_fn(eval_inputs, params),
                            lambda x: decode_target_ids(x, params),
                            params.output,
                            config,
                            device_list=params.device_list,
                            max_to_keep=params.keep_top_checkpoint_max,
                            eval_secs=params.eval_secs,
                            eval_steps=params.eval_steps
                        ),
                        step=params.update_cycle
                    )
                )
            checkpoint_dir = params.output
        else:
            checkpoint_dir = None


        ## easy session for a quick check-out
        # sess = tf.Session(config=config)
        # sess.run(tf.global_variables_initializer())
        # sess.run(restore_variables(args.checkpoint))
        # np.set_printoptions(threshold=np.inf) 
        # for i in range(1000000):
        #    a, b = sess.run([train_op, loss])
        #    print(b)

        restore_op = restore_variables(args.checkpoint)
        def restore_fn(step_context):
            step_context.session.run(restore_op)
        ## original version
        # Create session, do not use default CheckpointSaverHook
        with tf1.train.MonitoredTrainingSession(
              checkpoint_dir=checkpoint_dir,
              hooks=train_hooks,
              save_checkpoint_secs=None,
              config=config
            ) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)
            while not sess.should_stop():
                if TEST_INFERENCE:
                    a, b = sess.run([train_op, test_feature])
                    print(np.argmax(b, axis=-1))
                else:
                    sess.run(train_op)

if __name__ == "__main__":
    main(parse_args())