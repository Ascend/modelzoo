import six
import tensorflow as tf
import horovod.tensorflow as hvd



class DistributeOptimizer(tf.train.Optimizer):
    """ LARC implementation
        -------------------
        Parameters:
          - optimizer:     initial optimizer that you wanna apply
                           example: tf.train.MomentumOptimizer
          - learning_rate: initial learning_rate from initial optimizer
          - clip:          if True apply LARC otherwise LARS
          - epsilon:       default value is weights or grads are 0.
          - name
          - use_locking
    """

    def __init__(self, 
                 optimizer, 
                 iter_size=1.0, 
                 accum_dtype=tf.float16,
                 name="DistributeOptimizer", 
                 use_locking=False):
        super(DistributeOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._iter_size =  iter_size  
        self._accum_dtype=accum_dtype

        var_list = (
                tf.trainable_variables() +
                tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self._grads_accum = [ tf.Variable( tf.cast(tf.zeros_like(v.initialized_value()), self._accum_dtype), dtype=self._accum_dtype, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES] ) for v in var_list ]

        
    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)


    def apply_gradients(self, gradvars, *args, **kwargs):
        #---------- iter_size --------------
        global_step = tf.train.get_global_step()

        with tf.device('/cpu:0'):
            skip_update_ph = tf.equal( tf.floormod( global_step, tf.cast(self._iter_size, tf.int64)), 0)

        def itersize_op():
            grads_and_vars_accum = []
            accum_ops = []
            for i, (grad, var) in enumerate(gradvars):
                if grad is not None:
                    grad_accum = self._grads_accum[i]
                    grad = tf.cast(grad, self._accum_dtype)
                    add_grads = grad_accum + grad
                    accum_ops.append( tf.assign(grad_accum, add_grads) )
                    grads_and_vars_accum.append( (grad_accum, var) )
            accum_op = tf.group(accum_ops)

            with tf.control_dependencies([accum_op]):
                def update_and_clear_op():
                    processed_grads_and_vars = self.post_process_grads(grads_and_vars_accum) # post_process_grads allreduce, scale_iter_size
                    update_weight_op = self._optimizer.apply_gradients( processed_grads_and_vars) 
                    with tf.control_dependencies([update_weight_op]):
                        reinit_accum_gradient_op = tf.group( [ tf.assign(g, tf.zeros_like(g)) for g,v in grads_and_vars_accum ] )
                    return reinit_accum_gradient_op
                grad_updates = tf.cond( pred=skip_update_ph, true_fn=update_and_clear_op, false_fn=lambda:tf.group(tf.assign_add(global_step, tf.constant(1, dtype=tf.int64))) )
            return grad_updates

        def no_itersize_op():
            grads_and_vars_clean = []
            for grad, var in gradvars:
                if grad is not None:
                    grads_and_vars_clean.append( (grad, var) )

            processed_grads_and_vars = self.post_process_grads(grads_and_vars_clean) # post_process_grads allreduce

            update_weight_op_1 = self._optimizer.apply_gradients( processed_grads_and_vars) 

            return update_weight_op_1 

        if tf.contrib.framework.is_tensor(self._iter_size):  #used if batchsize scaling, in which case itersize is a tensor
            apply_gradients_op = tf.cond(tf.equal(self._iter_size, 1), no_itersize_op, itersize_op)
        else:
            apply_gradients_op = no_itersize_op() if float(self._iter_size)==1.0 else itersize_op()

        return apply_gradients_op

    def post_process_grads(self, grads_and_vars):

        dense_grads_and_vars = []
        for grad,var in grads_and_vars:
          if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                from tensorflow.python.training.optimizer import _deduplicate_indexed_slices
                summed_values, unique_indices = _deduplicate_indexed_slices(
                    values=grad.values, indices=grad.indices)
                gradient_no_duplicate_indices = tf.IndexedSlices(
                    indices=unique_indices,
                    values=summed_values,
                    dense_shape=grad.dense_shape)
                grad = tf.convert_to_tensor(gradient_no_duplicate_indices)
            dense_grads_and_vars.append((grad, var))

        g_list = [ g for g,v in dense_grads_and_vars ]
        #g_list_average = tensor_fusion_and_allreduce( g_list, hvd, small_thres=2500000, max_group=80 )
        g_list_average = [ hvd.allreduce(g, average=False) for g in g_list ]
        g_list_average = [ tf.cast(g, tf.float32) for g in g_list_average ]

        scale_factor = tf.cast(tf.multiply(self._iter_size, hvd.size()), tf.float32)
        grads_and_vars = [ ( tf.multiply(g_list_average[i], tf.cast( (1.0/scale_factor), tf.float32)), v) for i,(g,v) in enumerate(dense_grads_and_vars) ]

        return grads_and_vars
