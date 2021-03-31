import tensorflow as tf
import os

from npu_bridge.hccl import hccl_ops
from hccl.manage.api import get_rank_id
from hccl.manage.api import get_rank_size

from .mix_parallel_init import get_model_parallel_group, get_model_parallel_world_size, get_model_parallel_rank,get_data_parallel_rank


WORLD_SIZE = get_model_parallel_world_size()
RANK_INDEX = get_model_parallel_rank()

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# --------------------------------------------------------
@tf.RegisterGradient('grad_for_copy_to_model_parallel_region')
def copy_to_model_parallel_grad(op, grad):

    group = get_model_parallel_group()
    grad_new = hccl_ops.allreduce(grad, "sum", fusion=0, group = group)

    return grad_new

def copy_to_model_parallel_region( x ):
    with tf.get_default_graph().gradient_override_map( {"Identity":'grad_for_copy_to_model_parallel_region'} ):
      x_new = tf.identity(x, name='Identity')
    return x_new

# --------------------------------------------------------
@tf.RegisterGradient('grad_for_reduce_from_model_parallel_region')
def reduce_from_model_parallel_grad(op, grad):
    grad_new = tf.identity(grad)
    return grad_new
       
def reduce_from_model_parallel_region( x ):
    with tf.get_default_graph().gradient_override_map( {"HcomAllReduce":'grad_for_reduce_from_model_parallel_region'} ):
        group = get_model_parallel_group()
        x_new = hccl_ops.allreduce(x, "sum", fusion=0, group=group)
    return x_new

# --------------------------------------------------------
@tf.RegisterGradient('grad_for_gather_from_model_parallel_region')
def gather_from_model_parallel_grad(op, grad):
    grad_new = tf.identity(grad)
    return grad_new
       
def gather_from_model_parallel_region( x ):
    x_shape = shape_list(x)
    assert len(x_shape) >=1

    with tf.get_default_graph().gradient_override_map( {"HcomAllReduce":'grad_for_gather_from_model_parallel_region'} ):
        x_new = Allgather(x, dim = len(x_shape) - 1 ,group = get_model_parallel_group())
    return x_new


# --------------------------------------------------------
@tf.RegisterGradient('grad_for_scatter_to_model_parallel_region')
def scatter_to_model_parallel_grad(op, grad):
    grad_shape = shape_list(grad)
    grad_new = Allgather(grad, dim = len(grad_shape)-1 ,group = get_model_parallel_group())
    return grad_new
       
def scatter_to_model_parallel_region( x ): 
    x_shape = shape_list(x)
    with tf.get_default_graph().gradient_override_map( {"Split":'grad_for_scatter_to_model_parallel_region'} ):
        x_new = tf.split(x, WORLD_SIZE, axis = len(x_shape) - 1 )     #这里split之后不用取对应的切分后的grad?
    return x_new




# --------------------------------------------------------
@tf.RegisterGradient('Allgather_grad_for_smdp')
def allgather_grad(op, grad):
    print ('--- inputs of op input 0:', op.inputs[0])
    print ('--- inputs of op input 1 :', op.inputs[1])
    print ('--- inputs of op input 2 :', op.inputs[2])
    print ('--- inputs of op input 3 :', op.inputs[3])
    print ('--- inputs of op input 4 :', op.inputs[4])

    with tf.name_scope('cccccc_backward'):
      grad = grad * 11
    grad = tf.split( grad, WORLD_SIZE, axis = op.inputs[ 4 ] )
    print ('--- grad for allgather grad:', grad.append(None))
    print ('--- grad for allgather gradall:', grad)
    return grad

def Allgather( x, dim = 1 ,group=None, fusion=0, fusion_id=None):
    all_local_var = []
    for gpu_id in range(get_rank_size(group=group) ):
      if gpu_id == get_rank_id(group=group):
        all_local_var.append( x )
      else:
        all_local_var.append( tf.zeros_like(x) )
    all_vars = tf.concat( all_local_var, dim )

#    with tf.get_default_graph().gradient_override_map( {"ConcatV2":'Allgather_grad_for_smdp'} ):
#      x_new = tf.concat( x, axis = dim, name='Concat' )

    if fusion == 0:
      x_new = hccl_ops.allreduce(all_vars, "sum", fusion=0, group=group)
    else:
      x_new = hccl_ops.allreduce(all_vars, "sum", fusion=fusion, fusion_id=fusion_id, group=group)
    print ('--- x new:', x_new)
    return x_new









