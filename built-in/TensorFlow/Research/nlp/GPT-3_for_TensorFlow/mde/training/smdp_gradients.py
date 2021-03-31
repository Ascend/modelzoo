import tensorflow as tf
from . import recompute_grads


def compute_gradients(total_loss, recompute=False, var_partition=False):
    grads = recompute_grads.gradients( total_loss, tf.trainable_variables(), checkpoints='self_define' ) 
    grads_convert = []
    for g in grads:
        if g is not None:
            if isinstance( g, tf.IndexedSlices ):
                g = tf.convert_to_tensor(g)
            grads_convert.append(g)   

    grads = grads_convert
    grads_and_vars = list(zip( grads, tf.trainable_variables() ))

    return grads_and_vars




