import six
import tensorflow as tf

class Optimizer: 
    def __init__(self, config):
        self.config = config 

    def get_lbs_optimizer(self, opt):  #TODO input is ( self, hyper_param )

        # opt = LargeBatchSizeOptimizer(opt, weight_decay=self.config['weight_decay'], 
        #                                    accum_dtype = self.config['dtype'],
        #                                    use_lars = self.config['use_lars'],
        #                                    bn_lr_scale = self.config.get('bn_lr_scale', 1.0)
        #                                 )
        #opt = MixedPrecisionOptimizer(opt, self.config) 

        return opt

class MixedPrecisionOptimizer(tf.train.Optimizer):
    """An optimizer that updates trainable variables in fp32."""

    def __init__(self, optimizer, config):
        super(MixedPrecisionOptimizer, self).__init__(
            optimizer._use_locking,
            optimizer._name + '-MP',
        )
        self._optimizer = optimizer
        self._config = config
        loss_scale=self._config['loss_scale']
        self._loss_scale = float(loss_scale)
        self._fp32_to_fp16 = {}

        var_list = (
                tf.trainable_variables() +
                tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self.var_fp32_copy = [ tf.Variable( tf.cast(v.initialized_value(), tf.float32), 
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

        grads_and_vars_fp16 = self._optimizer.compute_gradients(
            loss, var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        # creating FP-32 variables and filling the fp32 dict
        grads_and_vars_fp32 = []

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

        grads_and_vars_fp32_rescaled = [ (g/self._loss_scale, v)  for g,v in grads_and_vars_fp32 ]


        return grads_and_vars_fp32_rescaled

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        update_op = self._optimizer.apply_gradients(grads_and_vars, *args, **kwargs)
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


class LargeBatchSizeOptimizer(tf.train.Optimizer):
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

    def __init__(self, optimizer, weight_decay, clip=True, epsilon=1., accum_dtype=tf.float16, use_lars=True, bn_lr_scale=1.0,
                 name="LarcOptimizer", use_locking=False):
        super(LargeBatchSizeOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
      #  self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._clip = clip
        self._epsilon = float(epsilon)
        self._accum_dtype=accum_dtype
        self._use_lars=use_lars
        self._bn_lr_scale=bn_lr_scale 

        var_list = (
                tf.trainable_variables() +
                tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        with tf.device('/gpu:0'):
            self._grads_accum = [ tf.Variable( tf.cast(tf.zeros_like(v.initialized_value()), self._accum_dtype), dtype=self._accum_dtype, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES] ) for v in var_list ]

        
    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)


    def apply_gradients(self, gradvars, loss_scale, *args, **kwargs):

        global_step = tf.train.get_global_step()

        grads_and_vars_clean = []
        for grad, var in gradvars:
            if grad is not None:
                grads_and_vars_clean.append( (grad, var) )

        processed_grads_and_vars = self.post_process_grads(grads_and_vars_clean, loss_scale) # post_process_grads includes Lars

        def apply():
            red_grad_updates = self._optimizer.apply_gradients( processed_grads_and_vars, global_step=tf.train.get_global_step() ) 
            return tf.group(red_grad_updates)

        update_weight_op_1 = apply()
        return update_weight_op_1 

        apply_gradients_op = update_weight_op_1

        with tf.device('/cpu:0'):
            #tf.summary.scalar('loss_scale', loss_scale)
            for grad, var in gradvars:
                g = grad / loss_scale
                v_norm_2 = tf.norm(var, ord='euclidean')
                g_norm_2 = tf.norm(g, ord='euclidean')
                v_g_norm2_ratio = v_norm_2 / (
                        g_norm_2 + self._weight_decay * v_norm_2)
                if grad is not None:
                    if 'BatchNorm' in var.name:
                        with tf.name_scope('bn_norm2/'):
                            tf.summary.scalar(var.name + '/norm2',
                                              v_norm_2)
                        with tf.name_scope('grad_bn_norm2/'):
                            tf.summary.scalar(var.name + '/grad_norm2',
                                              g_norm_2)
                        with tf.name_scope('bn_ratio_var_grad/'):
                            tf.summary.scalar(var.name + '/ratio_var_grad',
                                              v_g_norm2_ratio)
                    else:
                        with tf.name_scope('conv_norm2/'):
                            tf.summary.scalar(var.name + '/norm2',
                                              v_norm_2)
                        with tf.name_scope('grad_conv_norm2/'):
                            tf.summary.scalar(var.name + '/grad_norm2',
                                              g_norm_2)
                        with tf.name_scope('conv_ratio_var_grad/'):
                            tf.summary.scalar(var.name + '/ratio_var_grad',
                                              v_g_norm2_ratio)

        return apply_gradients_op

    def post_process_grads(self, grads_and_vars, loss_scale):

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
