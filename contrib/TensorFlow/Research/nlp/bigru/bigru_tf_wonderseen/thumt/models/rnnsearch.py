# coding=utf-8
#!/usr/bin/env python
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

import copy
import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils
from thumt.models.model import NMTModel
from thumt.parameter_config import *

 
def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)
 

def _gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None, max_length=60):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]

    
    if using_dynamic:
        time_steps = tf.shape(inputs)[1]
    else: ## static version
        time_steps = inputs.shape[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    if using_dynamic:
        input_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="input_array") # dymanic
        output_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="output_array") # dymanic
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))


        def loop_func(t, out_ta, state): # dymanic
            inp_t = input_ta.read(t) # dymanic
            cell_output, new_state = cell(inp_t, state)
            cell_output = _copy_through(t, sequence_length, zero_output,
                                        cell_output)
            new_state = _copy_through(t, sequence_length, state, new_state)
            out_ta = out_ta.write(t, cell_output) # dymanic
            return t + 1, out_ta, new_state
        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, initial_state)
        outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                                loop_vars, parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        final_state = outputs[2]
        all_output = output_final_ta.stack()
        all_output = tf.transpose(all_output, [1, 0, 2])
        return all_output, final_state
    else:
        input_ta = tf.transpose(inputs, [1, 0, 2], name="input_array")
        output_ta = []
        t = 0
        state = initial_state

        while t < max_length:
            inp_t = input_ta[t]
            cell_output, new_state = cell(inp_t, state)
            cell_output = _copy_through(t, sequence_length, zero_output, cell_output)
            state = _copy_through(t, sequence_length, state, new_state)
            output_ta.append(state)
            t = t + 1
        final_state = new_state
        output_final_ta = output_ta
        all_output = tf.stack(output_final_ta, axis=1, name="output_array")
        return all_output, final_state



def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None, max_length=60):
    with tf.variable_scope(scope or "encoder", values=[inputs, sequence_length]):
        inputs_fw = inputs

        inputs_bw = tf.reverse_sequence(inputs, sequence_length, batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None,
                                               dtype=dtype, max_length=max_length)

        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None,
                                               dtype=dtype, max_length=max_length)
            output_bw = tf.reverse_sequence(output_bw, sequence_length, batch_axis=0, seq_axis=1)
        

        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _decoder(cell, inputs, memory, sequence_length, initial_state, dtype=None,
             scope=None, mode=None, max_length=61):
    # Assume that the underlying cell is GRUCell-like
    if using_dynamic:
        time_steps = tf.shape(inputs)[1]
    else:
        time_steps = inputs.shape[1]


    batch = tf.shape(inputs)[0]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)


    if using_dynamic:
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"], maxlen=tf.shape(memory)[1], dtype=dtype)

        bias = layers.attention.attention_bias(mem_mask, "masking", dtype=dtype)
        bias = tf.squeeze(bias, axis=[1, 2])
        
        cache = layers.attention.attention(None, memory, None, output_size)

        input_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="input_array")
        output_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="output_array")
        value_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)
        initial_state = layers.nn.linear(initial_state, output_size, True, False, scope="s_transform")
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, att_ta, val_ta, state, cache_key):
            inp_t = input_ta.read(t)
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context]
            cell_output, new_state = cell(cell_input, state)
            cell_output = _copy_through(t, sequence_length["target"], zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state, new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value, context)


            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cache_key = tf.identity(cache_key)
            return t + 1, out_ta, att_ta, val_ta, new_state, cache_key

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state,
                     cache["key"])

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        if using_dynamic:
            final_output.set_shape([None, None, output_size])
        else:
            final_output.set_shape([time_steps, None, output_size])

        final_output = tf.transpose(final_output, [1, 0, 2])

        final_value = value_final_ta.stack()
        if using_dynamic:
            final_value.set_shape([None, None, memory.shape[-1].value])
        else:
            final_value.set_shape([time_steps, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])
    else:
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"], maxlen=tf.shape(memory)[1], dtype=dtype)

        bias = layers.attention.attention_bias(mem_mask, "masking", dtype=dtype)
        bias = tf.squeeze(bias, axis=[1, 2])
        
        cache = layers.attention.attention(None, memory, None, output_size)

        initial_state = layers.nn.linear(initial_state, output_size, True, False, scope="s_transform")
        initial_state = tf.tanh(initial_state)

        output_ta, alpha_ta, value_ta = [], [], []
        state = initial_state
        cache_key = cache["key"]
        t = 0
        while t < max_length-1: # not to predict the first token
            inp_t = inputs[t]
            results = layers.attention.attention(state, memory, bias,
                            output_size, cache={"key": cache_key}, reuse=tf.AUTO_REUSE)
            alpha = results["weight"]
            context = results["value"]
            cell_input = [inp_t, context]
            cell_output, new_state = cell(cell_input, state)

            cell_output = _copy_through(t, sequence_length["target"], zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state, new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value, context)

            output_ta.append(cell_output)
            alpha_ta.append(alpha)
            value_ta.append(new_value)

            cache_key = tf.identity(cache_key)
            state = new_state
            t += 1

        final_output = tf.stack(output_ta, axis=1, name="output_array")
        final_value = tf.stack(value_ta, axis=1, name="value_array")

    result = {
        "outputs": final_output,
        "values": final_value,
        "initial_state": initial_state
    }
    return result



def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    import collections
    import re
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def model_graph(features, mode, params, msame=False):
    if mode == "infer":
        return model_infer_graph(features, mode, params, msame=msame)
    else:
        return model_training_graph(features, mode, params)

def model_training_graph(features, mode, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])
    dtype = tf.get_variable_scope().dtype

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding", [src_vocab_size, params.embedding_size]) 
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding", [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])


    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)


    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size, reuse=tf.AUTO_REUSE, input_type=2)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size, reuse=tf.AUTO_REUSE, input_type=2)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=dtype
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=dtype
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs, features["source_length"],
        dtype=dtype, max_length=params.train_encode_length)

    # decoder
    cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size, input_type=3)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=dtype
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }

    initial_state = encoder_output["final_states"]["backward"]


    with tf.variable_scope("decoder", dtype=dtype):
        ## Shift left
        ## RNN style
        # shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
        # shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

        ## Transformer style
        shifted_tgt_inputs = tgt_inputs[:, :-1, :]
        
        ## Decoder
        decoder_output = _decoder(cell, shifted_tgt_inputs, encoder_output["annotation"],
                              length, initial_state, dtype=dtype, mode=mode,
                              max_length=params.train_decode_length)

        ## Shift output
        ## RNN style
        # all_outputs = tf.concat(
        #     [
        #         tf.expand_dims(decoder_output["initial_state"], axis=1),
        #         decoder_output["outputs"],
        #     ],
        #     axis=1
        # )
        # shifted_outputs = all_outputs[:, :-1, :]
        ## Transformer style
        shifted_outputs = decoder_output["outputs"]

        maxout_features = [
            shifted_tgt_inputs,
            shifted_outputs,
            decoder_output["values"]
        ]
        maxout_size = params.hidden_size // params.maxnum
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum, concat=False)
        readout = layers.nn.linear(maxhid, params.embedding_size, False, False, scope="deepout")
        if params.dropout and not params.use_variational_dropout:
            readout = tf.nn.dropout(readout, 1.0 - params.dropout)

        ## Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False, scope="softmax")


    ## labels
    # labels = features["target"] # RNN style
    labels = features["target"][:, 1:] # Transformer style
    logits = tf.reshape(logits, [-1, tgt_vocab_size])
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"]-1,
            maxlen=tf.shape(labels)[1]
        )
    )

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    return loss, logits

def model_infer_graph(features, mode, params, msame=False):
    if msame: # for testing msame, the batch size should be set 1
        for key, value in features.items():
            if params.eval_batch_size != 1:
                params.eval_batch_size = 1
                features[key] = value[:1]

    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])
    dtype = tf.get_variable_scope().dtype

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding", [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding", [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)

    ## encoder
    cell_fw = layers.rnn_cell.LegacyGRUCell(params.hidden_size, reuse=tf.AUTO_REUSE, input_type=2)
    cell_bw = layers.rnn_cell.LegacyGRUCell(params.hidden_size, reuse=tf.AUTO_REUSE, input_type=2)
    encoder_output = _encoder(cell_fw, cell_bw, src_inputs, features["source_length"], dtype=dtype, max_length=params.eval_encode_length)

    ## decoder
    cell = layers.rnn_cell.LegacyGRUCell(params.hidden_size, input_type=3)
    length = {
        "source": features["source_length"],
        "target": features["target_length"] if "target_length" in features else -1
    }
    sequence_length = length

    initial_state = encoder_output["final_states"]["backward"]
    memory = encoder_output["annotation"]
    output_size = cell.output_size

    with tf.variable_scope("decoder", dtype=dtype, reuse=tf.AUTO_REUSE):
        mem_mask = tf.sequence_mask(sequence_length["source"], maxlen=tf.shape(memory)[1], dtype=dtype)
        bias = layers.attention.attention_bias(mem_mask, "masking", dtype=dtype)
        bias = tf.squeeze(bias, axis=[1, 2])
        cache = layers.attention.attention(None, memory, None, output_size)
        initial_state = layers.nn.linear(initial_state, output_size, True, False, scope="s_transform")
        initial_state = tf.tanh(initial_state)
        
        state = initial_state
        all_outputs = initial_state
        cache_key = cache["key"]
        t = 0

        ## uncessary for NPU
        # zero_value = tf.zeros([EVAL_BATCH_SIZE, memory.shape[-1].value], dtype)

        target_outputs = []

        if "target" in features:
            if len(features["target"].get_shape().as_list()) == 2:
                target = features["target"][:,0]
            else:
                target = features["target"]
        else:
            target = tf.ones([params.eval_batch_size], dtype=tf.int32)*BOS_ID
        
        ## Here is a non-incremental decoding
        while t < params.eval_decode_length-1:
            ## RNN style
            # if t == 0:
            #     inp_t = tf.zeros([EVAL_BATCH_SIZE, params.embedding_size], dtype)
            # else:
            #     tgt_inputs = tf.nn.embedding_lookup(tgt_emb, target)
            #     inp_t = tf.nn.bias_add(tgt_inputs, tgt_bias)
            ## Transformer style
            tgt_inputs = tf.nn.embedding_lookup(tgt_emb, target)
            inp_t = tf.nn.bias_add(tgt_inputs, tgt_bias)

            ## uncessary for NPU
            # inp_t = tf.reshape(inp_t, [EVAL_BATCH_SIZE, params.embedding_size])
            
            results = layers.attention.attention(state, memory, bias, output_size, cache={"key": cache_key}, reuse=tf.AUTO_REUSE)
            alpha = results["weight"]
            context = results["value"]
            
            ## uncessary for NPU
            # context = tf.reshape(context, [EVAL_BATCH_SIZE, 2*output_size])
            # state = tf.reshape(state, [EVAL_BATCH_SIZE, output_size])

            cell_input = [inp_t, context]
            cell_output, new_state = cell(cell_input, state)
            new_value = context

            ## uncessary for NPU
            # new_value = _copy_through(t, sequence_length["target"], zero_value, context)

            cache_key = tf.identity(cache_key)
            state = new_state
            all_outputs = cell_output

            maxout_features = [
                inp_t,
                all_outputs,
                new_value
            ]


            maxout_size = params.hidden_size // params.maxnum
            maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum, concat=False)
            readout = layers.nn.linear(maxhid, params.embedding_size, False, False, scope="deepout")
            logit = layers.nn.linear(readout, tgt_vocab_size, True, False, scope="softmax", reuse=tf.AUTO_REUSE)
            target = tf.argmax(logit, axis=-1)
            target_outputs.append(tf.nn.log_softmax(logit))
            t += 1

        target_outputs = tf.stack(target_outputs, axis=1)

    return target_outputs


class RNNsearch(NMTModel):

    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearch, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "infer" if TEST_INFERENCE else "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self, msame=False):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                log_prob = model_graph(features, "infer", params, msame=msame)

            return log_prob

        return inference_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<sos>",
            append_eos=False,
            # model
            rnn_cell="LegacyGRUCell",
            embedding_size=620,
            hidden_size=1000,
            maxnum=2,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            clip_grad_norm=5.0
        )

        return params
