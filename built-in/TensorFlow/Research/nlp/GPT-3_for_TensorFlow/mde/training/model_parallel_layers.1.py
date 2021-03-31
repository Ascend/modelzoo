import tensorflow as tf
import numpy as np
import math
import os

from npu_bridge.hccl import hccl_ops 

from mde.distribute import dist
from mde.distribute.mix_parallel_init import get_model_parallel_world_size

#**********************************************
# this file includes:
# -- ColumnParallelLinear
# -- RowParallelLinear
# -- ParallelAtten
# -- ParallelMLP
#**********************************************

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def ColumnParallelLinear(x, output_size, gather_output=False):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
#    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
#        w_init_stdev = w_init_stdev * (1. / math.sqrt(2.0* params["n_layer"]))

    x_shape = shape_list(x)
    f_in = x_shape[-1]
    f_out = output_size
    w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )

    with tf.name_scope('ColumnParallelLinear'):
        world_size = get_model_parallel_world_size()
        print('11111111111111111111111111mode_parallel_size', world_size)
        assert output_size % world_size == 0, 'outputsize{} is not divisible by world_size{}'.format(output_size, world_size)
        output_size_partial = output_size//world_size

        x = dist.copy_to_model_parallel_region(x)  
        x = tf.layers.dense( inputs=x, units=output_size_partial, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=1) )
        x = gelu(x)

        if gather_output:
            output = dist.gather_from_model_parallel_region(x)
        else:
            output = x
        return output


def RowParallelLinear(x, output_size, input_is_parallel=True):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
#    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
#        w_init_stdev = w_init_stdev * (1. / math.sqrt(2.0* params["n_layer"]))


    world_size = get_model_parallel_world_size()

    x_shape = shape_list(x)
    f_in = x_shape[-1]

    if input_is_parallel:
        f_in = f_in * world_size

    f_out = output_size
    w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )

    with tf.name_scope('RowParallelLinear'):
     #   assert input_size % world_size == 0, 'input_size{} is not divisible by world_size{}'.format(input_size, world_size)
     #   input_size_partial = input_size//world_size

        if not input_is_parallel: #输入x没有在卡上切分，需要先对输入切分
            x = dist.scatter_to_model_parallel_region(x)

        x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=1))
        x = gelu(x)
    
        output = dist.reduce_from_model_parallel_region(x)
        return output









