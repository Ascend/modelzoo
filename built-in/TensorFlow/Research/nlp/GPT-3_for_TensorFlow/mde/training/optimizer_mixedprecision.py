import six
import tensorflow as tf
import horovod.tensorflow as hvd
from . import automatic_loss_scaler

class MixedPrecisionOptimizer(tf.train.Optimizer):
    """used if there exist two piece of variables in mix training ."""

    def __init__(self,
                 optimizer, 
                 config):
        super(MixedPrecisionOptimizer, self).__init__(
            optimizer._use_locking,
            optimizer._name + '-MP',
        )
        self._optimizer = optimizer
        self._config = config
        loss_scale = self._config['loss_scale']

        self._loss_scaler = None
        if isinstance(loss_scale, six.string_types):
            self._loss_scaler = automatic_loss_scaler.AutomaticLossScaler(algorithm=loss_scale, params=self._config)
            self._loss_scale = self._loss_scaler.loss_scale
        else:
            self._loss_scale = float(loss_scale)

        self._fp32_to_fp16 = {}

        var_list = (
                tf.trainable_variables() +
                tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self.var_fp32_copy = [ tf.Variable( tf.cast(tf.zeros_like(v.initialized_value()), tf.float32), 
                                    dtype=tf.float32, trainable=False, 
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "FP32_MASTER_COPIES"] ) for v in var_list ]


    def compute_gradients(self, loss, var_list=None,
                            gate_gradients=tf.train.Optimizer.GATE_OP,
                            aggregation_method=None,
                            colocate_gradients_with_ops=False,
                            grad_loss=None):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        if self._loss_scale != 1.0:
            loss = tf.scalar_mul(self._loss_scale, loss)
        with tf.name_scope('compute_the_gradient'):
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
          
        init_params_op = tf.cond( tf.equal( tf.train.get_global_step() ,0), init_param, lambda: tf.no_op())

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

        rescaled_grads_and_vars_fp32 = []
        for grad, var in grads_and_vars_fp32:
            rescaled_grads_and_vars_fp32.append( (grad/ self._loss_scale, var) )

        return rescaled_grads_and_vars_fp32

    def apply_gradients(self, grads_and_vars, *args, **kwargs):

        def apply():
            update_op = self._optimizer.apply_gradients(grads_and_vars,*args, **kwargs)
            return tf.group(update_op)

        if self._loss_scaler:
            grad_has_nans, grad_amax = automatic_loss_scaler.AutomaticLossScaler.check_grads(grads_and_vars) 
            should_skip_update = tf.logical_or(tf.is_inf(grad_amax), grad_has_nans)
            loss_scale_update_op = self._loss_scaler.update_op(grad_has_nans,grad_amax)
            with tf.control_dependencies([loss_scale_update_op]):
                update_weight_op = tf.cond(should_skip_update, lambda:tf.group(tf.assign_add(tf.train.get_global_step(), tf.constant(1, dtype=tf.int64))), apply)                        
        else:
            update_weight_op = apply() 

        apply_ops = []
        with tf.control_dependencies([update_weight_op]):
            for grad, var in grads_and_vars:
                if var.name in self._fp32_to_fp16:
                    dst_var = self._fp32_to_fp16[var.name]
                    apply_ops.append(
                        tf.assign(dst_var, tf.saturate_cast(var, tf.float16)))
        if apply_ops:
            return tf.group(apply_ops)
        return update_weight_op


class MixedPrecisionOptimizer_version2(tf.train.Optimizer):
    """used if mix training by getter"""

    def __init__(self, 
                 optimizer,
                 config):
        super(MixedPrecisionOptimizer_version2, self).__init__(
            optimizer._use_locking,
            optimizer._name + '-MP', 
        )
        self._config = config
        loss_scale = self._config['loss_scale']
        self._optimizer = optimizer

        self._loss_scaler = None
        if isinstance(loss_scale, six.string_types):
            self._loss_scaler = automatic_loss_scaler.AutomaticLossScaler(algorithm=loss_scale, params=self._config)
            self._scale = self._loss_scaler.loss_scale
        else:
            self._scale = float(loss_scale)
        

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        replaced_list = var_list

        self._scale = tf.identity(self._scale, name='loss_scale') 

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, replaced_list, *args, **kwargs)

        final_gradvar = []

        for orig_var, (grad, var) in zip(var_list, gradvar): 
            if var is not orig_var:
                grad = tf.cast(grad, orig_var.dtype)
            if self._scale != 1.0:
                grad = tf.scalar_mul(1. / self._scale, grad)
            final_gradvar.append((grad, orig_var))

        return final_gradvar

    def apply_gradients(self,grads_and_vars, *args, **kwargs):
        def apply():
            update_op = self._optimizer.apply_gradients(grads_and_vars,*args, **kwargs)
            return tf.group(update_op)

        if self._loss_scaler:
            grad_has_nans, grad_amax = automatic_loss_scaler.AutomaticLossScaler.check_grads(grads_and_vars) 
            should_skip_update = tf.logical_or(tf.is_inf(grad_amax), grad_has_nans) 
            loss_scale_update_op = self._loss_scaler.update_op(grad_has_nans,grad_amax)
            with tf.control_dependencies([loss_scale_update_op]):
                update_weight_op = tf.cond(should_skip_update, lambda:tf.group(tf.assign_add(tf.train.get_global_step(), tf.constant(1, dtype=tf.int64))), apply)                        
        else:
            update_weight_op = apply() 

        return update_weight_op





