import tensorflow as tf
from ..distribute import dist 
import numpy as np
import math

from mde.distribute.mix_parallel_init import get_model_parallel_world_size, get_model_parallel_rank

MODEL_PARALLEL_RANK = get_model_parallel_rank()
#**********************************************
# this file includes:
# -- ColumnParallelLinear
# -- RowParallelLinear
# -- ParallelAtten
# -- ParallelMLP
#**********************************************
def _initialize_affine_weight(input_size, output_size,
                              num_split, partition_dim, w_init_stdev, var_name=None): 
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    assert var_name is not None

    if get_model_parallel_world_size() == 1:
        master_weight = tf.get_variable(var_name, [input_size, output_size], initializer=tf.random_normal_initializer(stddev=w_init_stdev,seed=1), trainable=True)
        return master_weight
    else:
        master_weight = tf.random_normal([input_size, output_size], stddev=w_init_stdev, seed=1)

    weight_list = tf.split(master_weight, num_split, axis=partition_dim)
    partial_weight = weight_list[get_model_parallel_rank()]
    per_weight = tf.get_variable(var_name, initializer=partial_weight, trainable=True)
    return per_weight

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def ColumnParallelLinear(args, scope, x, output_size, gather_output=False, init_scale_by_depth=False):
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
    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
        w_init_stdev = 0.02 * (1. / math.sqrt(2.0* args["n_layer"]))
    else:
        w_init_stdev = 0.02

    # x_shape = shape_list(x)
    # f_in = x_shape[-1]
    # f_out = output_size
    # w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )
    
    with tf.variable_scope(scope):
        assert output_size % get_model_parallel_world_size() == 0, 'outputsize{} is not divisible by world_size{}'.format(output_size, get_model_parallel_world_size())
        output_size_partial = output_size//get_model_parallel_world_size()

        x = dist.copy_to_model_parallel_region(x) 
        x = tf.layers.dense( inputs=x, units=output_size_partial, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=MODEL_PARALLEL_RANK))
     #   x = tf.layers.dense( inputs=x, units=output_size_partial, kernel_initializer = tf.random_uniform_initializer() )
        # x = tf.layers.dense( inputs=x, units=output_size_partial, kernel_initializer = tf.constant_initializer(0.01) )
        #分布式初始化
        # w = _initialize_affine_weight(input_size=shape_list(x)[-1], 
        #                                 output_size=output_size,
        #                                 num_split=get_model_parallel_world_size(), 
        #                                 partition_dim=1, 
        #                                 w_init_stdev=w_init_stdev, 
        #                                 var_name='w')
        # b = tf.get_variable('b', [output_size//get_model_parallel_world_size()], initializer=tf.constant_initializer(0))
        # x = tf.matmul(x, w) + b

        # x = tf.Print(x,[x],message='fc1 output!!!!!!!!!!!!',summarize=-1)
        #######
        if gather_output:
            output = dist.gather_from_model_parallel_region(x)
        else:
            output = x
        return output


def RowParallelLinear(args, scope, x, output_size, input_is_parallel=True, init_scale_by_depth=False):
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
    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
        w_init_stdev = 0.02 * (1. / math.sqrt(2.0* args["n_layer"]))
    else:
        w_init_stdev = 0.02

    # x_shape = shape_list(x)
    # f_in = x_shape[-1]

    # if input_is_parallel:
    #     f_in = f_in * hvd.size()
 
    # f_out = output_size
    # w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )

    with tf.variable_scope(scope):

        if not input_is_parallel: #输入x没有在卡上切分，需要先对输入切分
            x = dist.scatter_to_model_parallel_region(x)

        x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=MODEL_PARALLEL_RANK))
 #       x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_uniform_initializer() )
        # x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.constant_initializer(0.01))
        #分布式初始化

        # w = _initialize_affine_weight(input_size=shape_list(x)[-1]*get_model_parallel_world_size() if input_is_parallel else shape_list(x)[-1], 
        #                                 output_size=output_size,
        #                                 num_split=get_model_parallel_world_size(), 
        #                                 partition_dim=0, 
        #                                 w_init_stdev=w_init_stdev, 
        #                                 var_name='w')
        # b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0))
        # x = tf.matmul(x, w) + b

        # x = tf.Print(x,[x],message='fc2 output!!!!!!!!!!!!' ,summarize=-1)
        ######

        output = dist.reduce_from_model_parallel_region(x)
        return output

#def ParallelEmbedding(x):
    
#-------------------------------------------------------------------------------------------------------


def ColumnLinear(args, scope, x, output_size, gather_output=False, init_scale_by_depth=False): 
    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
        w_init_stdev = 0.02 * (1. / math.sqrt(2.0* args["n_layer"]))
    else:
        w_init_stdev = 0.02

    # x_shape = shape_list(x)
    # f_in = x_shape[-1]
    # f_out = output_size
    # w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )

    with tf.variable_scope(scope):
        x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=MODEL_PARALLEL_RANK))
      #  x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_uniform_initializer() )
        # x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.constant_initializer(0.01) )
        return x


def RowLinear(args, scope, x, output_size, input_is_parallel=True, init_scale_by_depth=False):
    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
        w_init_stdev = 0.02 * (1. / math.sqrt(2.0* args["n_layer"]))
    else:
        w_init_stdev = 0.02
        
    # x_shape = shape_list(x)
    # f_in = x_shape[-1]

    # f_out = output_size
    # w_init_stdev=math.sqrt( 2.0 / float( f_in + f_out ) )

    with tf.variable_scope(scope):

        x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_normal_initializer(stddev=w_init_stdev,seed=MODEL_PARALLEL_RANK))
     #   x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.random_uniform_initializer() )
        # x = tf.layers.dense( inputs=x, units=output_size, kernel_initializer = tf.constant_initializer(0.01))
        return x






