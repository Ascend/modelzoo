import tensorflow as tf
import horovod.tensorflow as hvd
from ..distribute.dist import SMDP_Allgather

def _get_smdp_getter():

  def inner_smdp_getter(getter, name, shape=None, trainable=True, regularizer=None, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']   # fp16 default

    if requested_dtype == tf.float16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True

    # --------------- SMDP function ------------
    if trainable:
        need_allgather = False
        split_dim = 0
        split_shape = [ s for s in shape ]
        
        if 'embedding' in name:
            need_allgather = False
        else:
            for i, shape_each in enumerate( shape ):
                if shape_each % hvd.size() == 0:
                    split_shape[i] = int ( int( shape_each ) / hvd.size() )
                    need_allgather = True
                    split_dim = i
                    break
                else:
                    need_allgather = False

    print ('--- in smdp getter, name and shape:', name, shape, split_shape, need_allgather, split_dim)

    var = getter(name, split_shape, *args, **kwargs)    # create split variables
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    if cast_to_float16:
      var = math_ops.cast(var, tf.float16)

    # -- all gather variables ---
    if trainable:
        if need_allgather:
            var = SMDP_Allgather( var, split_dim )

    return var

  return inner_smdp_getter


