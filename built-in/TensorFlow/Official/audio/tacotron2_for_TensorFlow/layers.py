# -*-coding:utf8-*-

import tensorflow as tf
import numpy as np
from tensorflow.nn import rnn_cell
from tensorflow.python.framework import constant_op
import os
from npu_bridge.estimator import npu_ops
from tensorflow.compat.v2 import while_loop
from tensorflow.python.compat import v2_compat
from tensorflow.python.ops import functional_ops
from tensorflow.python.framework import function
from tensorflow.python.ops import gen_array_ops

# v2_compat.enable_v2_behavior()
# tf.compat.v1.enable_control_flow_v2()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_resource_variables()

ks = tf.keras


def cell_compare(inputs, units):
    input_dim = int(inputs.shape[-1])
    batch_size = int(inputs.shape[0])
    weight = np.random.rand(input_dim + units, 4 * units).astype(np.float32) - 0.5
    npu_weight = tf.Variable(initial_value=weight, dtype=tf.float32)
    bias = np.zeros(shape=[4 * units], dtype=np.float32)
    npu_bias = tf.Variable(initial_value=bias, dtype=tf.float32)
    weight_initializer = tf.initializers.constant(value=weight, dtype=tf.float32)
    nn_cell = rnn_cell.LSTMCell(num_units=units, initializer=weight_initializer)
    init_hidden, init_state = nn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    nn_h, nn_c = nn_cell(inputs, [init_hidden, init_state])
    ct, ht, it, jt, ft, ot, tanhct = npu_ops.basic_lstm_cell(
            inputs,
            init_hidden,
            init_state,
            npu_weight,
            npu_bias,
            keep_prob=1.0,
            forget_bias=1.0,
            state_is_tuple=True,
            activation='tanh'
        )
    return nn_h, ht


class Embedding(object):
    """
    embedding layer implementation on NPU
    use nn operations to replace keras layers
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.built = False

    def build(self, input_shape):
        # check shape OK
        init_value = (np.random.rand(self.input_dim, self.output_dim) - 0.5) / 10
        self.embedding = tf.Variable(
            shape=(self.input_dim, self.output_dim),
            initial_value=init_value,
            name='embedding',
            trainable=True,
            dtype=tf.float32
        )
        self.built = True

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        inputs = tf.cast(inputs, dtype=tf.int32)
        input_ids = tf.one_hot(inputs, depth=self.input_dim)
        out_embed = seq_matmul(inputs=input_ids, weights=self.embedding)
        return out_embed


class NPULSTMCell(object):
    """
    keras style NPU implementation of LSTMCell
    wraps the kernel and other variables
    """
    def __init__(self,
                 units,
                 kernel_initializer=None,
                 bias_initializer=None,
                 forget_bias=1.0,
                 **kwargs):
        super(NPULSTMCell, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias

        # add statement of other output of npu_lstm_cell
        self.it = None
        self.jt = None
        self.ft = None
        self.ot = None
        self.tanhct = None

        self.built = False
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # do not support self.add_weight
        kernel = np.random.rand(input_dim + self.units, 4 * self.units).astype(np.float32) - 0.5
        bias = np.zeros(shape=[4 * self.units, ], dtype=np.float32)
        self.kernel = tf.Variable(initial_value=kernel,
                                  dtype=tf.float32,
                                  trainable=True,
                                  name='npu_lstm_kernel')
        self.bias = tf.Variable(initial_value=bias,
                                dtype=tf.float32,
                                trainable=True,
                                name='npu_lstm_kernel')
        self.built = True

    def zero_state(self, batch_size, dtype):
        return tf.zeros(shape=[batch_size, self.units], dtype=dtype), \
               tf.zeros(shape=[batch_size, self.units], dtype=dtype)

    def __call__(self, inputs, states, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        hidden, cell = states
        ct, ht, self.it, self.jt, self.ft, self.ot, self.tanhct = npu_ops.basic_lstm_cell(
            inputs,
            hidden,
            cell,
            self.kernel,
            self.bias,
            keep_prob=1.0,
            forget_bias=self.forget_bias,
            state_is_tuple=True,
            activation='tanh'
        )
        return ht, [ht, ct]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = int(inputs.shape[0])
            dtype = inputs.dtype
        init_hidden = tf.zeros(shape=[batch_size, self.units], dtype=dtype)
        init_cell = tf.zeros(shape=[batch_size, self.units], dtype=dtype)
        return [init_hidden, init_cell]

    @property
    def state_size(self):
        return self.units, self.units

    @property
    def output_size(self):
        return self.units


class NPUBiLSTM(object):
    """
        kears style NPU implementation of bi-LSTM
    """
    def __init__(self,
                 units,
                 kernel_initializer=None,
                 bias_initializer=None,
                 forget_bias=1.0,
                 slice_len=48,
                 return_sequence=True,
                 **kwargs):
        super(NPUBiLSTM, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias
        self.slice_len = slice_len
        self.forward_cell = NPULSTMCell(units=self.units,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer,
                                        forget_bias=self.forget_bias)
        self.backward_cell = NPULSTMCell(units=self.units,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer,
                                         forget_bias=self.forget_bias)
        self.return_sequence = return_sequence
        # self.forward_cell = rnn_cell.LSTMCell(num_units=self.units, initializer=self.kernel_initializer)
        # self.backward_cell = rnn_cell.LSTMCell(num_units=self.units, initializer=self.kernel_initializer)
        self.built = False

    def build(self, input_shape):
        self.forward_cell.build(input_shape=input_shape)
        self.backward_cell.build(input_shape=input_shape)
        self.built = True

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        reverse_inputs = tf.reverse(inputs, axis=[1])
        batch_size = int(inputs.shape[0])
        seq_len = int(inputs.shape[1])
        input_dim = int(inputs.shape[-1])
        fw_init_hid, fw_init_cell = self.forward_cell.get_initial_state(batch_size=batch_size, dtype=inputs.dtype)
        bw_init_hid, bw_init_cell = self.backward_cell.get_initial_state(batch_size=batch_size, dtype=inputs.dtype)
        start = 0
        if self.return_sequence:
            output_sequence = tf.TensorArray(size=seq_len,
                                             dtype=inputs.dtype,
                                             element_shape=[batch_size, 1, self.units * 2])

            def cond(idx, fw_hid, fw_cell, bw_hid, bw_cell, out_seq):
                return idx < seq_len

            def body(idx, fw_hid, fw_cell, bw_hid, bw_cell, out_seq):
                # fw_frame = inputs[:, idx, :]
                fw_frame = tf.squeeze(tf.slice(inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
                # bw_frame = reverse_inputs[:, idx, :]
                bw_frame = tf.squeeze(tf.slice(reverse_inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
                fw_out_hid, fw_out_state = self.forward_cell(fw_frame, [fw_hid, fw_cell])
                bw_out_hid, bw_out_state = self.backward_cell(bw_frame, [bw_hid, bw_cell])
                out_frame = tf.expand_dims(tf.concat([fw_out_hid, bw_out_hid], axis=-1), axis=1)
                out_seq = out_seq.write(idx, out_frame)
                idx += 1
                return idx, fw_out_hid, fw_out_state[1], bw_out_hid, bw_out_state[1], out_seq

            out_idx, out_fw_hid, out_fw_cell, out_bw_hid, out_bw_cell, output_sequence = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[
                    start,
                    fw_init_hid,
                    fw_init_cell,
                    bw_init_hid,
                    bw_init_cell,
                    output_sequence
                ]
            )
            output_list = [output_sequence.read(i) for i in range(seq_len)]
            # output_slices = self.make_slice(output_list)
            # output = gen_array_ops.concat(concat_dim=1, values=output_list)
            output = tf.concat(output_list, axis=1)
            return output
        else:

            def cond(idx, fw_hid, fw_cell, bw_hid, bw_cell):
                return idx < seq_len

            def body(idx, fw_hid, fw_cell, bw_hid, bw_cell):
                fw_frame = tf.squeeze(tf.slice(inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
                bw_frame = tf.squeeze(tf.slice(reverse_inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
                fw_out_hid, fw_out_state = self.forward_cell(fw_frame, [fw_hid, fw_cell])
                bw_out_hid, bw_out_state = self.backward_cell(bw_frame, [bw_hid, bw_cell])
                idx += 1
                return idx, fw_out_hid, fw_out_state[1], bw_out_hid, bw_out_state[1]

            out_idx, out_fw_hid, out_fw_cell, out_bw_hid, out_bw_cell = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[
                    start,
                    fw_init_hid,
                    fw_init_cell,
                    bw_init_hid,
                    bw_init_cell,
                ]
            )

            return tf.concat([out_fw_hid, out_bw_hid], axis=-1)

    def make_slice(self, work_list):
        start = 0
        slices = []
        while start < len(work_list):
            if start + self.slice_len > len(work_list):
                slices.append(gen_array_ops.concat(concat_dim=1, values=work_list[start:]))
            else:
                slices.append(gen_array_ops.concat(concat_dim=1, values=work_list[start:start + self.slice_len]))
            start += self.slice_len
        return slices


class NPUDropout(object):
    """
    NPU implementation of dropout
    """
    def __init__(self,
                 keep_prob,
                 **kwargs):
        super(NPUDropout, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.built = False

    def __call__(self, inputs, **kwargs):
        dropped = npu_ops.dropout(inputs, keep_prob=self.keep_prob)
        return dropped


class NNConv1D(object):
    """
    nn implementation of conv1d
    """
    def __init__(self,
                 kernel_size,
                 strides,
                 filters,
                 use_bias=True,
                 padding='SAME',
                 activation='linear',
                 **kwargs):
        super(NNConv1D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.filters = filters
        self.activation = activation

        self.built = False

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = [self.kernel_size, input_dim, self.filters]
        init_kernel = np.random.rand(self.kernel_size, input_dim, self.filters) - 0.5
        self.kernel = tf.Variable(initial_value=init_kernel,
                                  dtype=tf.float32,
                                  trainable=True,
                                  name='conv1d_kernel')
        if self.use_bias:
            init_bias = np.zeros(shape=[self.filters, ], dtype=np.float32)
            self.bias = tf.Variable(initial_value=init_bias,
                                    dtype=tf.float32,
                                    trainable=True,
                                    name='conv1d_bias')
        else:
            self.bias = None
        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        conv_out = tf.nn.conv1d(value=inputs,
                                filters=self.kernel,
                                padding=self.padding,
                                stride=self.strides,
                                use_cudnn_on_gpu=False,
                                data_format='NWC')
        if self.use_bias:
            conv_out = conv_out + self.bias
        if self.activation == 'tanh':
            conv_out = tf.nn.tanh(conv_out)
        elif self.activation == 'relu':
            conv_out = tf.nn.relu(conv_out)
        elif self.activation == 'sigmoid':
            conv_out = tf.nn.sigmoid(conv_out)
        return conv_out


class NNBatchNorm(object):
    """
    batch normalization with tf.nn API
    """
    def __init__(self,
                 axis=-1,
                 scale=True,
                 offset=True,
                 moment=0.99,
                 epsilon=1e-3,
                 **kwargs):
        super(NNBatchNorm, self).__init__(**kwargs)
        self.axis = axis
        self.scale = scale
        self.offset = offset
        self.moment = moment
        self.epsilon = epsilon

        self.built = False

    def build(self, input_shape):
        input_dim = input_shape[self.axis]
        init_mean = np.zeros(shape=[input_dim], dtype=np.float32)
        self.moving_mean = tf.Variable(initial_value=init_mean,
                                       dtype=tf.float32,
                                       trainable=False,
                                       name='batch_norm_moving_mean')
        init_var = np.ones(shape=[input_dim], dtype=np.float32)
        self.moving_var = tf.Variable(initial_value=init_var,
                                      dtype=tf.float32,
                                      trainable=False,
                                      name='batch_norm_moving_var')
        if self.scale:
            init_gamma = np.ones(shape=[input_dim], dtype=np.float32)
            self.gamma = tf.Variable(initial_value=init_gamma,
                                     dtype=tf.float32,
                                     trainable=True,
                                     name='batch_norm_gamma')
        else:
            self.gamma = None
        if self.offset:
            init_beta = np.zeros(shape=[input_dim], dtype=np.float32)
            self.beta = tf.Variable(initial_value=init_beta,
                                    dtype=tf.float32,
                                    trainable=True,
                                    name='batch_norm_beta')
        else:
            self.beta = None
        self.built = True

    def __call__(self, inputs, training=True, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        rank = len(inputs.shape)
        reduction_axes = [i for i in range(rank) if i != self.axis]
        if training:
            mean, variance = tf.nn.moments(inputs, reduction_axes)
            update_mean = tf.compat.v1.assign(
                self.moving_mean, tf.identity(self.moving_mean) * self.moment + (1 - self.moment) * mean)
            update_var = tf.compat.v1.assign(
                self.moving_var, tf.identity(self.moving_var) * self.moment + (1 - self.moment) * variance)
            with tf.control_dependencies([update_mean, update_var]):
                normed_batch = tf.nn.batch_normalization(inputs,
                                                         mean=mean,
                                                         variance=variance,
                                                         offset=self.beta,
                                                         scale=self.gamma,
                                                         variance_epsilon=self.epsilon)
                return normed_batch
        else:
            mean, variance = self.moving_mean, self.moving_var
            normed_batch = tf.nn.batch_normalization(inputs,
                                                     mean=mean,
                                                     variance=variance,
                                                     offset=self.beta,
                                                     scale=self.gamma,
                                                     variance_epsilon=self.epsilon)
            return normed_batch


class NNDense(object):
    """
    Dense layer use tf.matmul to replace keras
    """
    def __init__(self,
                 units,
                 use_bias=True,
                 activation='linear',
                 **kwargs):
        super(NNDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation

        self.built = False

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        init_kernel = np.random.rand(input_dim, self.units) - 0.5
        self.kernel = tf.Variable(initial_value=init_kernel,
                                  dtype=tf.float32,
                                  trainable=True,
                                  name='dense_kernel')
        if self.use_bias:
            init_bias = np.zeros(shape=[self.units], dtype=np.float32)
            self.bias = tf.Variable(initial_value=init_bias,
                                    dtype=tf.float32,
                                    trainable=True,
                                    name='dense_bias')
        else:
            self.bias = None
        self.built = True

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        rank = len(inputs.shape)
        if rank == 2:
            output = tf.matmul(inputs, self.kernel)
        elif rank == 3:
            output = ks.backend.dot(inputs, self.kernel)
        else:
            output = None
        # output = ks.backend.dot(inputs, self.kernel)

        if self.use_bias:
            output = output + self.bias

        if self.activation == 'tanh':
            output = tf.nn.tanh(output)
        elif self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        return output


class BasicRNN(object):
    """
    Basic RNN layer use tf.matmul to replace keras
    """
    def __init__(self,
                 units,
                 use_bias=True,
                 activation='linear',
                 **kwargs):
        super(BasicRNN, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation

        self.built = False

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        init_kernel = np.random.rand(input_dim + self.units, self.units) - 0.5
        self.kernel = tf.Variable(initial_value=init_kernel,
                                  dtype=tf.float32,
                                  trainable=True,
                                  name='dense_kernel')
        if self.use_bias:
            init_bias = np.zeros(shape=[self.units], dtype=np.float32)
            self.bias = tf.Variable(initial_value=init_bias,
                                    dtype=tf.float32,
                                    trainable=True,
                                    name='dense_bias')
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        self.__call__(inputs=inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)

        batch_size = int(inputs.shape[0])
        seq_len = int(inputs.shape[1])
        input_dim = int(inputs.shape[-1])
        initial_state = tf.zeros(shape=[batch_size, self.units])
        start = 0

        def cond(idx, hidden):
            return idx < seq_len

        def body(idx, hidden):
            input_slice = tf.slice(inputs, [0, idx, 0], [batch_size, 1, input_dim])
            input_slice = tf.squeeze(input_slice, axis=1)
            input_hidden = tf.concat([input_slice, hidden], axis=-1)
            out_hidden = tf.matmul(input_hidden, self.kernel)
            if self.use_bias:
                out_hidden = out_hidden + self.bias
            idx += 1
            return idx, out_hidden

        _, output = tf.while_loop(cond=cond, body=body, loop_vars=[start, initial_state])

        if self.activation == 'tanh':
            output = tf.nn.tanh(output)
        elif self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        return output


class NPUStaticBiLSTM(object):
    """
        kears style NPU implementation of bi-LSTM
    """
    def __init__(self,
                 units,
                 kernel_initializer=None,
                 bias_initializer=None,
                 forget_bias=1.0,
                 slice_len=48,
                 **kwargs):
        super(NPUStaticBiLSTM, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.forget_bias = forget_bias
        self.slice_len = slice_len
        self.forward_cell = NPULSTMCell(units=self.units,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer,
                                        forget_bias=self.forget_bias)
        self.backward_cell = NPULSTMCell(units=self.units,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer,
                                         forget_bias=self.forget_bias)
        self.built = False

    def build(self, input_shape):
        self.forward_cell.build(input_shape=input_shape)
        self.backward_cell.build(input_shape=input_shape)
        self.built = True

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(input_shape=inputs.shape)
        reverse_inputs = tf.reverse(inputs, axis=[1])
        batch_size = int(inputs.shape[0])
        seq_len = int(inputs.shape[1])
        input_dim = int(inputs.shape[-1])
        fw_init_hid, fw_init_cell = self.forward_cell.get_initial_state(batch_size=batch_size, dtype=inputs.dtype)
        bw_init_hid, bw_init_cell = self.backward_cell.get_initial_state(batch_size=batch_size, dtype=inputs.dtype)
        output_sequence = tf.TensorArray(size=seq_len,
                                         dtype=inputs.dtype,
                                         element_shape=[batch_size, 1, self.units * 2])

        for idx in range(seq_len):
            fw_frame = tf.squeeze(tf.slice(inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
            fw_init_hid, fw_cell = self.forward_cell(fw_frame, [fw_init_hid, fw_init_cell])
            fw_init_cell = fw_cell[1]
            bw_frame = tf.squeeze(tf.slice(reverse_inputs, [0, idx, 0], [batch_size, 1, input_dim]), axis=1)
            bw_init_hid, bw_cell = self.backward_cell(bw_frame, [bw_init_hid, bw_init_cell])
            bw_init_cell = bw_cell[1]
            out_frame = tf.expand_dims(tf.concat([fw_init_hid, bw_init_hid], axis=-1), axis=1)
            output_sequence = output_sequence.write(idx, out_frame)

        output_list = [output_sequence.read(i) for i in range(seq_len)]
        output = tf.concat(output_list, axis=1)
        return output

    def make_slice(self, work_list):
        start = 0
        slices = []
        while start < len(work_list):
            if start + self.slice_len > len(work_list):
                slices.append(gen_array_ops.concat(concat_dim=1, values=work_list[start:]))
            else:
                slices.append(gen_array_ops.concat(concat_dim=1, values=work_list[start:start + self.slice_len]))
            start += self.slice_len
        return slices


def seq_matmul(inputs, weights):
    print(weights.name)
    seq_len = int(inputs.shape[1])
    batch_size = int(inputs.shape[0])
    input_dim = int(inputs.shape[-1])
    out_dim = int(weights.shape[1])
    out_ta = tf.TensorArray(size=seq_len, dtype=inputs.dtype, element_shape=[batch_size, 1, out_dim])
    # out_ta = (out_ta, out_ta)
    start = 0

    # @function.Defun(tf.int32, tf.float32)
    def cond(idx, ta):
        return idx < seq_len

    # @function.Defun(tf.int32, tf.float32)
    def body(idx, ta):
        # in_frame = inputs[:, idx, :]
        in_frame = tf.slice(inputs, [0, idx, 0], [batch_size, 1, input_dim])
        in_frame = tf.squeeze(in_frame, axis=1)
        print(in_frame.shape, weights.shape, '\n\n')
        out_frame = tf.expand_dims(tf.matmul(in_frame, weights), axis=1)
        # print('in body: ', type(ta))
        ta = ta.write(idx, out_frame)
        # print(type(ta))
        idx += 1
        return idx, ta

    _, out_seq = tf.while_loop(cond=cond, body=body, loop_vars=[start, out_ta])
    # print("before", out_ta)
    # _, out_seq = functional_ops.While(cond=cond, body=body, input_=[start, out_ta])
    # print("after", out_seq)
    out_list = [out_seq.read(i) for i in range(seq_len)]
    # output = None
    # for i in range(seq_len):
    #     if output is None:
    #         frame = out_list[i]
    #         output = frame
    #     else:
    #         frame = out_list[i]
    #         output = gen_array_ops.concat(values=[output, frame], concat_dim=1)
    # output = gen_array_ops.concat(concat_dim=1, values=out_list)
    output = tf.concat(out_list, axis=1)
    return output
