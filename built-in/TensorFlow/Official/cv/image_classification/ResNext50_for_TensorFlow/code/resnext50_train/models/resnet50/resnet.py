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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf

_BATCH_NORM_EPSILON = 1e-4
_BATCH_NORM_DECAY = 0.9

_Cardi = 32


class LayerBuilder(object):
    def __init__(self, activation=None, data_format='channels_last',
                 training=False, use_batch_norm=False, batch_norm_config=None,
                 conv_initializer=None, bn_init_mode='adv_bn_init', bn_gamma_initial_value=1.0):
        self.activation = activation
        self.data_format = data_format
        self.training = training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self.conv_initializer = conv_initializer
        self.bn_init_mode = bn_init_mode
        self.bn_gamma_initial_value = bn_gamma_initial_value
        if self.batch_norm_config is None:
            self.batch_norm_config = {
                'decay': _BATCH_NORM_DECAY,
                'epsilon': _BATCH_NORM_EPSILON,
                'scale': True,
                'zero_debias_moving_mean': False,
            }

    def _conv2d(self, inputs, activation, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
          #  use_bias=not self.use_batch_norm,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None if self.use_batch_norm else activation,
            *args, **kwargs)
        if self.use_batch_norm:
            param_initializers = {
                'moving_mean': tf.zeros_initializer(),
                'moving_variance': tf.ones_initializer(),
                'beta': tf.zeros_initializer(),
            }
            if self.bn_init_mode == 'adv_bn_init':
                param_initializers['gamma'] = tf.ones_initializer()
            elif self.bn_init_mode == 'conv_bn_init':
                param_initializers['gamma'] = tf.constant_initializer(self.bn_gamma_initial_value)
            else:
                raise ValueError("--bn_init_mode must be 'conv_bn_init' or 'adv_bn_init' ")

            x = self.batch_norm(x)
            x = activation(x) if activation is not None else x
        return x

    def conv2d_linear_last_bn(self, inputs, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None, *args, **kwargs)
        param_initializers = {
            'moving_mean': tf.zeros_initializer(),
            'moving_variance': tf.ones_initializer(),
            'beta': tf.zeros_initializer(),
        }
        if self.bn_init_mode == 'adv_bn_init':
            param_initializers['gamma'] = tf.zeros_initializer()
        elif self.bn_init_mode == 'conv_bn_init':
            param_initializers['gamma'] = tf.constant_initializer(self.bn_gamma_initial_value)
        else:
            raise ValueError("--bn_init_mode must be 'conv_bn_init' or 'adv_bn_init' ")    

        x = self.batch_norm(x, param_initializers=param_initializers)
        return x

    def conv2d_no_act_no_bn(self, inputs, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None, *args, **kwargs)
        return x

    def conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, *args, **kwargs)

    def conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, *args, **kwargs)

    def pad2d(self, inputs, begin, end=None):
        if end is None:
            end = begin
        try:
            _ = begin[1]
        except TypeError:
            begin = [begin, begin]
        try:
            _ = end[1]
        except TypeError:
            end = [end, end]
        if self.data_format == 'channels_last':
            padding = [[0, 0], [begin[0], end[0]], [begin[1], end[1]], [0, 0]]
        else:
            padding = [[0, 0], [0, 0], [begin[0], end[0]], [begin[1], end[1]]]
        return tf.pad(inputs, padding)

    def max_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def average_pooling2d_stride_1(self, inputs, *args, **kwargs):
     #   inputs = tf.nn.avg_pool(inputs, ksize=[1,1,1,1],strides=[1,1,1,1], padding="VALID", data_format="NHWC" )
        return inputs

    def average_pooling2d(self, inputs, *args, **kwargs):
        inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding="VALID", data_format="NHWC" )
        return inputs

 #       return tf.layers.average_pooling2d(
 #           inputs, data_format=self.data_format, *args, **kwargs)

    def dense_linear(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=None)

    def dense(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=self.activation)

    def activate(self, inputs, activation=None):
        activation = activation or self.activation
        return activation(inputs) if activation is not None else inputs

    def batch_norm(self, inputs, **kwargs):
        all_kwargs = dict(self.batch_norm_config)
        all_kwargs.update(kwargs)
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        bn_inputs = inputs
        outputs = tf.contrib.layers.batch_norm(
            inputs, is_training=self.training, data_format=data_format,
            fused=True, **all_kwargs)

        return outputs

    def spatial_average2d(self, inputs):
        shape = inputs.get_shape().as_list()
        if self.data_format == 'channels_last':
            n, h, w, c = shape
        else:
            n, c, h, w = shape
        n = -1 if n is None else n
        x = tf.layers.average_pooling2d(inputs, (h, w), (1, 1),
                                        data_format=self.data_format)
        return tf.reshape(x, [n, c])

    def flatten2d(self, inputs):
        x = inputs
        if self.data_format != 'channel_last':
            # Note: This ensures the output order matches that of NHWC networks
            x = tf.transpose(x, [0, 2, 3, 1])
        input_shape = x.get_shape().as_list()
        num_inputs = 1
        for dim in input_shape[1:]:
            num_inputs *= dim
        return tf.reshape(x, [-1, num_inputs], name='flatten')

    def residual2d(self, inputs, network, units=None, scale=1.0, activate=False):
        outputs = network(inputs)
        c_axis = -1 if self.data_format == 'channels_last' else 1
        h_axis = 1 if self.data_format == 'channels_last' else 2
        w_axis = h_axis + 1
        ishape, oshape = [y.get_shape().as_list() for y in [inputs, outputs]]
        ichans, ochans = ishape[c_axis], oshape[c_axis]
        strides = ((ishape[h_axis] - 1) // oshape[h_axis] + 1,
                   (ishape[w_axis] - 1) // oshape[w_axis] + 1)
        with tf.name_scope('residual'):
            if (ochans != ichans or strides[0] != 1 or strides[1] != 1):
                inputs = self.conv2d_linear(inputs, units, 1, strides, 'SAME')
            x = inputs + scale * outputs
            if activate:
                x = self.activate(x)
        return x


def resnet_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride, filters, arch_type,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[3]
    x = inputs
    #with tf.name_scope('resnet_model'):
    if depth == num_inputs:
        if stride == 1:#v1.5
            shortcut = x
        else:#v1
            shortcut = builder.max_pooling2d(x, 1, stride)
    else: # the downsample(first) block in each layer
        if 'D1' in arch_type:
            if stride == 1:
                shortcut = builder.average_pooling2d_stride_1(x, stride, stride)             #--------------------Resnet-D------------
            else:
                shortcut = builder.average_pooling2d(x, stride, stride)             #--------------------Resnet-D------------
            shortcut = builder.conv2d_linear(shortcut, depth, 1, 1, 'SAME')
        elif 'D2' in arch_type:
            shortcut = builder.conv2d_linear(x, depth, 3, stride, 'SAME')
        elif 'D3' in arch_type:
            shortcut = builder.conv2d_linear(x, depth, 1, 1, 'SAME')
            shortcut = builder.average_pooling2d(shortcut, stride, stride)             #--------------------Resnet-D------------
        else:
            shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')
        conv_input = x

    if basic:
        x = builder.pad2d(x, 1)
        x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
        x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
    else:
        conv_input = x
        x = builder.conv2d(x, depth_bottleneck, 1, 1, 'SAME')
        conv_input = x
        if stride == 1:
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
        else:
            if 'E1' in arch_type:
                x = builder.average_pooling2d( x, stride, stride )
                x = builder.conv2d(x, depth_bottleneck, 3, 1, 'SAME')
            elif 'E2' in arch_type:
                x = builder.conv2d(x, depth_bottleneck, 3, 1, 'SAME')
                if stride == 1:
                    x = builder.average_pooling2d_stride_1(x, stride, stride)
                else:
                    x = builder.average_pooling2d(x, stride, stride)
            else:  # E0
                x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
            
        # x = builder.conv2d_linear(x, depth,            1, 1,      'SAME')
        conv_input = x
        x = builder.conv2d_linear_last_bn(x, depth, 1, 1, 'SAME')

    x = tf.nn.relu(x + shortcut)
    return x




def resnext_bottleneck(builder, inputs, depth, depth_bottleneck, stride, filters, arch_type,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[3]
    x = inputs
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:#v1.5
                shortcut = x
            else:#v1
                shortcut = builder.max_pooling2d(x, 1, stride)
        else: # the downsample(first) block in each layer
            shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
        else:

            #----- split layer ------
            x = builder.conv2d( x, depth_bottleneck, 1, 1, 'SAME' )   

            group_inputs = tf.split( x, _Cardi, axis=3 )

            layers_split=[]
            tmp = x
            for i in range(_Cardi):
                with tf.name_scope('cardi_' + str(i)):
                    split = builder.conv2d_no_act_no_bn(group_inputs[i], depth_bottleneck / _Cardi,
                                                        3, stride, 'SAME')
                    layers_split.append(split)

            x = tf.concat(layers_split, axis=3)
            x = builder.batch_norm(x)
            x = tf.nn.relu(x)

            x = builder.conv2d_linear_last_bn(x, depth, 1, 1, 'SAME')
        x = tf.nn.relu(x + shortcut)
        return x


def resnet_bottleneck_v2(builder, inputs, depth, depth_bottleneck, stride, filters, arch_type,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[1]
    x = inputs
    with tf.name_scope('resnet_v1'):
        # ------- shortcut ---------------
        if depth == num_inputs:
            if stride == 1:#v1.5
                shortcut = x
                x = builder.batch_norm(x)
                x = tf.nn.relu(x)
            else:#v1
                shortcut = builder.max_pooling2d(x, 1, stride)
        else: # the downsample(first) block in each layer
            x = builder.batch_norm(x)
            x = tf.nn.relu(x)

            if 'D1' in arch_type:
                shortcut = builder.average_pooling2d(x, stride, stride)             #--------------------Resnet-D------------
                shortcut = builder.conv2d_linear(shortcut, depth, 1, 1, 'SAME')
            elif 'D2' in arch_type:
                shortcut = builder.conv2d_linear(x, depth, 3, stride, 'SAME')
            elif 'D3' in arch_type:
                shortcut = builder.conv2d_linear(x, depth, 1, 1, 'SAME')
                shortcut = builder.average_pooling2d(shortcut, stride, stride)             #--------------------Resnet-D------------
            else:
                shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')

        # -------- mainstream ----------------
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
        else:
            x = builder.conv2d(x, depth_bottleneck, 1, 1, 'SAME')
            x = builder.batch_norm(x)
            x = tf.nn.relu(x)

            if stride == 1:
                x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
                x = builder.batch_norm(x)
                x = tf.nn.relu(x)
            else:
                if 'E1' in arch_type:
                    x = builder.average_pooling2d( x, stride, stride )
                    x = builder.conv2d(x, depth_bottleneck, 3, 1, 'SAME')
                    x = builder.batch_norm(x)
                    x = tf.nn.relu(x)
                elif 'E2' in arch_type:
                    x = builder.conv2d(x, depth_bottleneck, 3, 1, 'SAME')
                    x = builder.batch_norm(x)
                    x = tf.nn.relu(x)
                    x = builder.average_pooling2d( x, stride, stride )
                else:  # E0
                    x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
                    x = builder.batch_norm(x)
                    x = tf.nn.relu(x)
            
            x = builder.conv2d_linear(x, depth, 1, 1, 'SAME')

        x = x + shortcut
        return x


def inference_resnext_impl(builder, inputs, layer_counts, arch_type='C1+D', num_classes=1001, basic=False):
    x = inputs
    #x = builder.batch_norm(x)
    x = builder.pad2d(x, 3)
    x = builder.conv2d(x, 64, 7, 2, 'VALID')
    #x = builder.conv2d(x, 64, 7, 2, 'SAME')
    

    num_filters=64
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    #x, argmax = tf.nn.max_pool_with_argmax(input=x, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')

    for i in range(layer_counts[0]):
        x = resnext_bottleneck(builder, x, 256, 128, 1, num_filters, arch_type, basic)
    for i in range(layer_counts[1]):
        num_filters = num_filters * 2
        x = resnext_bottleneck(builder, x, 512, 256, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[2]):
        num_filters = num_filters * 2
        x = resnext_bottleneck(builder, x, 1024, 512, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[3]):
        num_filters = num_filters * 2
        x = resnext_bottleneck(builder, x, 2048, 1024, 2 if i == 0 else 1, num_filters, arch_type, basic)
    print('====================Final x:', x)

    axes = [1, 2]
    x = tf.reduce_mean(x, axes, keepdims=True)
    x = tf.identity(x, 'final_reduce_mean')
    x = tf.reshape(x, [-1, 2048])
    x = tf.layers.dense(inputs=x, units=num_classes, kernel_initializer=tf.variance_scaling_initializer())
    x = tf.identity(x, 'final_dense')
    return x       
        

def inference_resnet_v1_impl(builder, inputs, layer_counts, arch_type='C1+D', resnet_version='v1.5', basic=False):
    x = inputs
    #x = builder.pad2d(x, 1)

    if 'C1' in arch_type:  # --- Resnet C -----
        x = builder.conv2d(x, 32, 3, 2, 'SAME')
        x = builder.conv2d(x, 32, 3, 1, 'SAME')
        x = builder.conv2d(x, 64, 3, 1, 'SAME')
    elif 'C2' in arch_type:  
        x = builder.conv2d(x, 32, 3, 1, 'SAME')
        x = builder.conv2d(x, 32, 3, 2, 'VALID')
        x = builder.conv2d(x, 64, 3, 1, 'VALID')
    elif 'C3' in arch_type:  
        x = builder.conv2d(x, 32, 3, 1, 'VALID')
        x = builder.conv2d(x, 32, 3, 1, 'VALID')
        x = builder.conv2d(x, 64, 3, 2, 'VALID')
    else:
        x = builder.conv2d(x, 64, 7, 2, 'SAME')

    num_filters=64

    pooled_inputs = x
    #x = builder.max_pooling2d(x, 3, 2, 'SAME')
    x, argmax = tf.nn.max_pool_with_argmax(input=x, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')

    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v1(builder, x, 256, 64, 1, num_filters, arch_type, basic)
    for i in range(layer_counts[1]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 512, 128, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[2]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 1024, 256, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[3]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v1(builder, x, 2048, 512, 2 if i == 0 else 1, num_filters, arch_type, basic)

    axes = [1,2]
    x = tf.reduce_mean( x, axes, keepdims=True )		
    x = tf.identity(x, 'final_reduce_mean')
    x = tf.reshape( x, [-1, 2048] )
    x = tf.layers.dense(inputs=x, units=1001,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    x = tf.identity( x, 'final_dense' )
    return x


def inference_resnet_v2_impl(builder, inputs, layer_counts, arch_type='C1+D', basic=False):
    x = inputs
    x = builder.pad2d(x, 3)

    if 'C1' in arch_type:  # --- Resnet C -----
        x = builder.conv2d(x, 32, 3, 2, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 32, 3, 1, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 64, 3, 1, 'SAME')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
    elif 'C2' in arch_type:  
        x = builder.conv2d(x, 32, 3, 1, 'SAME')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 32, 3, 2, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 64, 3, 1, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
    elif 'C3' in arch_type:  
        x = builder.conv2d(x, 32, 3, 1, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 32, 3, 1, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
        x = builder.conv2d(x, 64, 3, 2, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)
    else:
        x = builder.conv2d(x, 64, 7, 2, 'VALID')
        x = builder.batch_norm(x)
        x = tf.nn.relu(x)

    num_filters=64

    pooled_inputs = x
    x = builder.max_pooling2d(x, 3, 2, 'SAME')

    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v2(builder, x, 256, 64, 1, num_filters, arch_type, basic)
    for i in range(layer_counts[1]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v2(builder, x, 512, 128, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[2]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v2(builder, x, 1024, 256, 2 if i == 0 else 1, num_filters, arch_type, basic)
    for i in range(layer_counts[3]):
        num_filters=num_filters*2
        x = resnet_bottleneck_v2(builder, x, 2048, 512, 2 if i == 0 else 1, num_filters, arch_type, basic)
    return builder.spatial_average2d(x)


def inference_resnet_v1(config, inputs, nlayer, data_format='channels_last',
                        training=False, conv_initializer=None, bn_init_mode='adv_bn_init', bn_gamma_initial_value=1.0):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    if config['resnet_version'] == 'v1.5':
        builder = LayerBuilder(tf.nn.relu, data_format, training, use_batch_norm=True,
                               conv_initializer=conv_initializer, bn_init_mode=bn_init_mode, bn_gamma_initial_value=bn_gamma_initial_value)
        if nlayer == 18:
            return inference_resnet_v1_impl(builder, inputs, [2, 2, 2, 2], config['arch_type'], config['resnet_version'], basic=True)
        elif nlayer == 34:
            return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'], config['resnet_version'], basic=True)
        elif nlayer == 50:
            return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'], config['resnet_version'])
        elif nlayer == 101:
            return inference_resnet_v1_impl(builder, inputs, [3, 4, 23, 3], config['arch_type'], config['resnet_version'])
        elif nlayer == 152:
            return inference_resnet_v1_impl(builder, inputs, [3, 8, 36, 3], config['arch_type'], config['resnet_version'])
        else:
            raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                             nlayer)

    elif config['resnet_version'] == 'v2':
        builder = LayerBuilder( None, data_format, training, use_batch_norm=False,
                               conv_initializer=conv_initializer, bn_init_mode=bn_init_mode, bn_gamma_initial_value=bn_gamma_initial_value)
        if nlayer == 18:
            return inference_resnet_v2_impl(builder, inputs, [2, 2, 2, 2], config['arch_type'], basic=True)
        elif nlayer == 34:
            return inference_resnet_v2_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'], basic=True)
        elif nlayer == 50:
            return inference_resnet_v2_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'])
        elif nlayer == 101:
            return inference_resnet_v2_impl(builder, inputs, [3, 4, 23, 3], config['arch_type'])
        elif nlayer == 152:
            return inference_resnet_v2_impl(builder, inputs, [3, 8, 36, 3], config['arch_type'])
        else:
            raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                             nlayer)
                             
    elif config['resnet_version'] == 'resnext':
        builder = LayerBuilder(tf.nn.relu, data_format, training, use_batch_norm=True,
                               conv_initializer=conv_initializer, bn_init_mode=bn_init_mode,
                               bn_gamma_initial_value=bn_gamma_initial_value)
        if nlayer == 18:
            return inference_resnext_impl(builder, inputs, [2, 2, 2, 2], config['arch_type'],
                                          config['num_classes'], basic=True)
        elif nlayer == 34:
            return inference_resnext_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'],
                                          config['num_classes'], basic=True)
        elif nlayer == 50:
            return inference_resnext_impl(builder, inputs, [3, 4, 6, 3], config['arch_type'], config['num_classes'])
        elif nlayer == 101:
            return inference_resnext_impl(builder, inputs, [3, 4, 23, 3], config['arch_type'], config['num_classes'])
        elif nlayer == 152:
            return inference_resnext_impl(builder, inputs, [3, 8, 36, 3], config['arch_type'], config['num_classes'])
        else:
            raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                             nlayer)                             
                             
                             
    else:
        raise ValueError("Invalid resnet version")
   


