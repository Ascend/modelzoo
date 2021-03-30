import tensorflow as tf
import horovod.tensorflow as hvd
#from smdp import constant


# ################################
# #SMDP getter
# ################################
def smdp_getter(getter, name, shape=None, dtype=None,
                          trainable=True, regularizer=None,
                          *args, **kwargs):
    dtype = tf.float16
    storage_dtype = tf.float32 if trainable else dtype
    print ('--- in smdp getter dytpe;', dtype)
    # ---- decide which dim to split -----
    shape_list = [s for s in shape]
    if trainable:
        need_allgather = False
        split_dim = 0
        split_shape = []

        if 'embedding' in name:
            need_allgather=False
        else:
            for i, shape_each in enumerate( shape_list ):
                if shape_each % hvd.size() == 0:
                    shape_list[i] = int(shape_each) / hvd.size()
                    need_allgather = True
                    split_dim = i 
                    break
                else:
                    need_allgather = False

        print ( 'Current Layer:', name,' is Split:', need_allgather, ' split dim:', split_dim, ' ori_shape:', shape, ' split shape:', shape_list )

    variable = getter(name, shape_list, dtype=storage_dtype,
                      trainable=trainable,
                      regularizer=regularizer if trainable and 'BatchNorm' not in name and 'batchnorm' not in name and 'batch_norm' not in name and 'Batch_Norm' not in name else None,
                      *args, **kwargs)

    if trainable and dtype != tf.float32:
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0')
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable


    if trainable:
        if need_allgather:

            @tf.custom_gradient
            def Allgather( var ):
                all_local_var = []
                for gpu_id in range( hvd.size() ):
                  if gpu_id == hvd.rank():
                    all_local_var.append( var )
                  else:
                    all_local_var.append( tf.zeros_like(var) )
                all_vars = tf.concat( all_local_var, split_dim )
                with tf.device('/cpu:0'):
                  all_gather_result = hvd.allreduce( all_vars, average=False )
            
                def grad(dy):
                    with tf.device('/cpu:0'):
                      avg_grad = hvd.allreduce( dy, average=True )
                    avg_grad_split = tf.split( avg_grad, num_or_size_splits=hvd.size(), axis=split_dim )
                    allgather_grad = avg_grad_split[ hvd.rank() ]
                    return allgather_grad
                return all_gather_result, grad
            
            variable = Allgather( variable )
            variable = tf.identity( variable, name='New_allgather_x_')
        else:
            variable = tf.identity( variable, name='New_x_')
        tf.add_to_collection('All_Gather', (variable, split_dim))

    return variable

#def all_gather(var, dim):
#    all_local_var = []
#    for gpu_id in range( hvd.size() ):
#      if gpu_id == hvd.rank():
#        all_local_var.append( var )
#      else:
#        all_local_var.append( tf.zeros_like(var) )
#    all_vars = tf.concat( all_local_var, dim )
#    result = Allgather_subfunc(all_vars)
#
##    all_vars = hvd.allreduce( all_vars,average=False )
#    return result
#
#
#@tf.custom_gradient
#def Allgather( var, dim ):
#    all_local_var = []
#    for gpu_id in range( hvd.size() ):
#      if gpu_id == hvd.rank():
#        all_local_var.append( var )
#      else:
#        all_local_var.append( tf.zeros_like(var) )
#    all_vars = tf.concat( all_local_var, dim )
#    all_gather_result = hvd.allreduce( all_vars, average=False )
#
#    def grad(dy):
#        avg_grad = hvd.allreduce( dy, average=True )
#        avg_grad_split = tf.split( avg_grad, num_or_size_splits=hvd.size(), axis=dim )
#        allgather_grad = avg_grad_split[hvd.rank()]
#        return allgather_grad
#    
#    return all_gather_result, grad




def float32_variable_storage_getter_ori(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
   # storage_dtype = tf.float32 if trainable else dtype
    storage_dtype = tf.float16
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if isinstance( variable, tf.IndexedSlices ):
        print ('== in getter, IndexSlices:', variable)

  #  if trainable and dtype != tf.float32:
  #      variable = tf.cast(variable, dtype)
    return variable


def _fp32_trainvar_getter(getter, name, shape=None, dtype=None,
                          trainable=True, regularizer=None,
                          *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      trainable=trainable,
                      regularizer=regularizer if trainable and 'BatchNorm' not in name and 'batchnorm' not in name and 'batch_norm' not in name and 'Batch_Norm' not in name else None,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0')
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable
    return variable


def fp32_trainable_vars(name='fp32_vars',var_partition=False, *args, **kwargs):
    """A varible scope with custom variable getter to convert fp16 trainable
    variables with fp32 storage followed by fp16 cast.
    """
    # return tf.variable_scope(
    #     name, custom_getter=_fp32_trainvar_getter, *args, **kwargs)
    if var_partition:
        return tf.variable_scope(
            name, custom_getter=smdp_getter, *args, **kwargs)
    else:
        return tf.variable_scope(
            name, custom_getter=float32_variable_storage_getter_ori, *args, **kwargs)

def custom_getter_with_fp16_and_weight_decay(dtype, weight_decay):
    return fp32_trainable_vars(dtype=dtype, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
