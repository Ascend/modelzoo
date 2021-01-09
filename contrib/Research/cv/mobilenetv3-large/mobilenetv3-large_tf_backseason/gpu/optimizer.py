import six
import tensorflow as tf
import horovod.tensorflow as hvd
from hvd_tensorfusion_sum import tensor_fusion_and_allreduce
from automatic_loss_scaler import AutomaticLossScaler

class Optimizer: 
    def __init__(self, config):
        self.config = config 

    def get_lbs_optimizer(self, opt, iter_size):  #TODO input is ( self, hyper_param )

        opt = LargeBatchSizeOptimizer(opt, weight_decay=self.config['weight_decay'], 
                                           iter_size = iter_size,
                                           accum_dtype = self.config['dtype'],
                                           use_lars = self.config['use_lars'],
                                           bn_lr_scale = self.config.get('bn_lr_scale', 1.0)
                                        )
#        opt = MixedPrecisionOptimizer(opt, self.config) 

        return opt

class MixedPrecisionOptimizer(tf.compat.v1.train.Optimizer):
    """An optimizer that updates trainable variables in fp32."""

    def __init__(self, optimizer, config):
        super(MixedPrecisionOptimizer, self).__init__(
            optimizer._use_locking,
            optimizer._name + '-MP',
        )
        self._optimizer = optimizer
        self._config = config
        loss_scale=self._config['loss_scale']

        self._loss_scaler = None
        if isinstance(loss_scale, six.string_types):
            self._loss_scaler = AutomaticLossScaler(algorithm=loss_scale, params=self._config)
            self._loss_scale = self._loss_scaler.loss_scale
        else:
            self._loss_scale = float(loss_scale)

        self._fp32_to_fp16 = {}

        var_list = (
                tf.trainable_variables() +
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self.var_fp32_copy = [ tf.Variable( tf.cast(tf.zeros_like(v.initialized_value()), tf.float32), 
                                    dtype=tf.float32, trainable=False, 
                                    collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "FP32_MASTER_COPIES"] ) for v in var_list ]


    def compute_gradients(self, loss, var_list=None,
                            gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                            aggregation_method=None,
                            colocate_gradients_with_ops=False,
                            grad_loss=None):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        if self._loss_scale != 1.0:
            loss = tf.scalar_mul(self._loss_scale, loss)

        grads_and_vars_fp16 = self._optimizer.compute_gradients(
            loss, var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        # creating FP-32 variables and filling the fp32 dict
        grads_and_vars_fp32 = []

        def init_param():
            return tf.group(*[ tf.assign( self.var_fp32_copy[i], tf.cast(v,tf.float32) ) for i, v in enumerate(var_list) ])
          
        init_params_op = tf.cond( tf.equal( tf.compat.v1.train.get_global_step() ,0), init_param, lambda: tf.no_op())

        with tf.control_dependencies([init_params_op]):
            with tf.variable_scope('FP32-master-copy'):
                for i, (grad, var) in enumerate(grads_and_vars_fp16):
                  if grad is not None:
                    if var.dtype.base_dtype == tf.float16:
                        fp32_var = self.var_fp32_copy[i]
                        self._fp32_to_fp16[fp32_var.name] = var
                        fp32_grad = tf.cast(grad, tf.float32)
                        grads_and_vars_fp32.append((fp32_grad, fp32_var))
                    else:
                        grads_and_vars_fp32.append((grad, var))
                  else:
                    grads_and_vars_fp32.append((None, var))

        # grads_and_vars_fp32 = _scale_grads(grads_and_vars_fp32,  # Here we remove the rescale to LargeBatchSize optimzier, in order to mimigrate the real 1024P transition.
        #                                    1.0 / self._loss_scale)
        return grads_and_vars_fp32

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        update_op = self._optimizer.apply_gradients(grads_and_vars, self._loss_scale, self._loss_scaler, *args, **kwargs)
        apply_ops = []
        with tf.control_dependencies([update_op]):
            for grad, var in grads_and_vars:
                if var.name in self._fp32_to_fp16:
                    dst_var = self._fp32_to_fp16[var.name]
                    apply_ops.append(
                        tf.assign(dst_var, tf.saturate_cast(var, tf.float16)))
        if apply_ops:
            return tf.group(apply_ops)
        return update_op


class LargeBatchSizeOptimizer(tf.compat.v1.train.Optimizer):
    """ LARC implementation
        -------------------
        Parameters:
          - optimizer:     initial optimizer that you wanna apply
                           example: tf.compat.v1.train.MomentumOptimizer
          - learning_rate: initial learning_rate from initial optimizer
          - clip:          if True apply LARC otherwise LARS
          - epsilon:       default value is weights or grads are 0.
          - name
          - use_locking
    """

    def __init__(self, optimizer, weight_decay, clip=True, epsilon=1., iter_size=1.0, accum_dtype=tf.float16, use_lars=True, bn_lr_scale=1.0,
                 name="LarcOptimizer", use_locking=False):
        super(LargeBatchSizeOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
      #  self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._clip = clip
        self._epsilon = float(epsilon)
        self._iter_size =  iter_size
        self._accum_dtype=accum_dtype
        self._use_lars=use_lars
        self._bn_lr_scale=bn_lr_scale 

        var_list = (
                tf.trainable_variables() +
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self._grads_accum = [ tf.Variable( tf.cast(tf.zeros_like(v.initialized_value()), self._accum_dtype), dtype=self._accum_dtype, trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES] ) for v in var_list ]

        
    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)


    def apply_gradients(self, gradvars, loss_scale, loss_scaler=None, *args, **kwargs):
        #---------- iter_size --------------
        global_step = tf.compat.v1.train.get_global_step()

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
                    processed_grads_and_vars = self.post_process_grads(grads_and_vars_accum, loss_scale) # post_process_grads includes Lars, allreduce, scale_iter_size

                    def apply():
                        red_grad_updates = self._optimizer.apply_gradients( processed_grads_and_vars, global_step=tf.compat.v1.train.get_global_step() ) 
                        return tf.group(red_grad_updates)

                    if loss_scaler:
                        grad_has_nans, grad_amax = AutomaticLossScaler.check_grads(processed_grads_and_vars) 
                        should_skip_update = tf.logical_or(tf.is_inf(grad_amax), grad_has_nans)
                        loss_scale_update_op = loss_scaler.update_op(grad_has_nans,
                                                                        grad_amax)
                        with tf.control_dependencies([loss_scale_update_op]):
                            update_weight_op = tf.cond(should_skip_update, lambda:tf.group(tf.assign_add( global_step, tf.constant(1, dtype=tf.int64))), apply)                        
                    else:
                        update_weight_op = apply() 
                    
                    with tf.control_dependencies([update_weight_op]):
                        reinit_accum_gradient_op = tf.group( [ tf.assign(g, tf.zeros_like(g)) for g,v in grads_and_vars_accum ] )
                    return reinit_accum_gradient_op
                grad_updates = tf.cond( pred=skip_update_ph, true_fn=update_and_clear_op, false_fn=lambda:tf.group(tf.assign_add( global_step, tf.constant(1, dtype=tf.int64))) )
            return grad_updates

        def no_itersize_op():
            grads_and_vars_clean = []
            for grad, var in gradvars:
                if grad is not None:
                    grads_and_vars_clean.append( (grad, var) )

            processed_grads_and_vars = self.post_process_grads(grads_and_vars_clean, loss_scale) # post_process_grads includes Lars, allreduce, scale_iter_size

            def apply():
                red_grad_updates = self._optimizer.apply_gradients( processed_grads_and_vars, global_step=tf.compat.v1.train.get_global_step() ) 
                return tf.group(red_grad_updates)

            if loss_scaler:
                grad_has_nans, grad_amax = AutomaticLossScaler.check_grads(processed_grads_and_vars) 
                should_skip_update = tf.logical_or(tf.is_inf(grad_amax), grad_has_nans)
                loss_scale_update_op = loss_scaler.update_op(grad_has_nans,
                                                                grad_amax)
                with tf.control_dependencies([loss_scale_update_op]):
                    update_weight_op_1 = tf.cond(should_skip_update, lambda:tf.group(tf.assign_add( global_step, tf.constant(1, dtype=tf.int64))), apply)                        
            else:
                update_weight_op_1 = apply()
            return update_weight_op_1 

        if tf.contrib.framework.is_tensor(self._iter_size):  #used if batchsize scaling, in which case itersize is a tensor
            apply_gradients_op = tf.cond(tf.equal(self._iter_size, 1), no_itersize_op, itersize_op)
        else:
            apply_gradients_op = no_itersize_op() if float(self._iter_size)==1.0 else itersize_op()

        return apply_gradients_op

    def post_process_grads(self, grads_and_vars, loss_scale):

        dense_grads_and_vars = []
        for grad,var in grads_and_vars:
          if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
             # if model._decoder.params.get('shared_embed', False):
                from tensorflow.python.training.optimizer import _deduplicate_indexed_slices
                summed_values, unique_indices = _deduplicate_indexed_slices(
                    values=grad.values, indices=grad.indices)
                gradient_no_duplicate_indices = tf.IndexedSlices(
                    indices=unique_indices,
                    values=summed_values,
                    dense_shape=grad.dense_shape)
                grad = tf.convert_to_tensor(gradient_no_duplicate_indices)
           # avg_grad = allreduce(grad)
            dense_grads_and_vars.append((grad, var))


        g_list = [ g for g,v in dense_grads_and_vars ]
        g_list_average = tensor_fusion_and_allreduce( g_list, hvd, small_thres=2500000, max_group=80 )
        g_list_average = [ tf.cast(g, tf.float32) for g in g_list_average ]
        scale_factor = tf.cast(tf.multiply(self._iter_size, hvd.size()), tf.float32)

        grads_and_vars = [ ( tf.multiply(g_list_average[i], tf.cast( (1.0/scale_factor), tf.float32)), v) for i,(g,v) in enumerate(dense_grads_and_vars) ]
        g_and_v_scaled = []
        for g, v in grads_and_vars:
            g = g / loss_scale
            g_and_v_scaled.append((g,v))

        # Lars
        if self._use_lars:
            grad_var_list = []
            #-----------------------------------------------LARS and weight decay-----------------------------------
            for g, var in  g_and_v_scaled:
                if 'BatchNorm' not in var.name and 'bias' not in var.name:
                    grad_norm = tf.norm(g,ord='euclidean') 
                    weight_norm = tf.norm(var,ord='euclidean')
                    
                    grad_norm_wd = tf.add( grad_norm,  tf.multiply( self._weight_decay, weight_norm ) )
                    rescale_factor = tf.div( tf.multiply(0.001, weight_norm), tf.add(grad_norm_wd, tf.constant(1e-5, tf.float32)) )

                    coeffi = tf.clip_by_value( rescale_factor, 0.001, 50.0 )
                    decayed_g = tf.add( g, tf.multiply( self._weight_decay, var ) )

                    g = tf.multiply(coeffi, decayed_g) 
                else:
                    g = self._bn_lr_scale * g

                g_and_v = ( g, var )
                grad_var_list.append( g_and_v )
            #-------------------------------------------LARS end---------------------------------
            return grad_var_list
        else:
            grad_var_list_without_lars = []
            #----------------------------------------weight decay-----------------------------------
            for g, var in  g_and_v_scaled:
                if 'BatchNorm' not in var.name and 'bias' not in var.name:
                    decayed_g = tf.add( g, tf.multiply( self._weight_decay, var ) )
                    g = decayed_g
                else:
                    g = self._bn_lr_scale * g

                g_and_v = ( g, var )
                grad_var_list_without_lars.append( g_and_v )

            return grad_var_list_without_lars
