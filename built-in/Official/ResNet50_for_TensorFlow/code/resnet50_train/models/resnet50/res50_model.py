
import tensorflow as tf
from . import resnet, res50_helper
from trainers.train_helper import stage
#from tensorflow.contrib.offline_train.python.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
#from tensorflow.contrib.offline_train.python import npu_ops
from npu_bridge.estimator import npu_ops
_NUM_EXAMPLES_NAME="num_examples"


class Model(object):
    def __init__(self, config, data, hyper_param, layers, optimizer, loss, logger):
        self.config = config
        self.data = data
        self.hyper_param = hyper_param
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.logger = logger  

    def get_estimator_model_func(self, features, labels, mode, params=None):
        labels = tf.reshape(labels, (-1,))  # Squash unnecessary unary dim         #----------------not use when use onehot label
    
        model_func = self.get_model_func()
        inputs = features  # TODO: Should be using feature columns?
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.device('/gpu:0'):
            if self.config['accelerator'] == 'gpu':
                inputs = tf.cast(inputs, self.config['dtype'])

            inputs = tf.cast(inputs, self.config['dtype'])
            with res50_helper.custom_getter_with_fp16_and_weight_decay(dtype=self.config['dtype'], weight_decay=self.config['weight_decay']):   # no BN decay

                top_layer = model_func(
                    inputs, data_format=self.config['data_format'], training=is_training,
                    conv_initializer=self.config['conv_init'],
                    bn_init_mode=self.config['bn_init_mode'], bn_gamma_initial_value=self.config['bn_gamma_initial_value'])
                

            logits = top_layer
            predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
            logits = tf.cast(logits, tf.float32)

            #loss = self.loss.get_loss(logits, labels)  
            #loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

            labels_one_hot = tf.one_hot(labels, depth=1001)
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels_one_hot, label_smoothing=self.config['label_smoothing'])


            base_loss = tf.identity(loss, name='loss')  # For access by logger (TODO: Better way to access it?)
     #       base_loss = tf.add_n([loss])                                    

            def exclude_batch_norm(name):
              #return 'batch_normalization' not in name
              return 'BatchNorm' not in name
            loss_filter_fn = exclude_batch_norm
          
            # Add weight decay to the loss.
            l2_loss = self.config['weight_decay'] * tf.add_n(
                # loss is computed using fp32 for numerical stability.
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                 if loss_filter_fn(v.name)])
            #tf.summary.scalar('l2_loss', l2_loss)
     #       total_loss = base_loss + l2_loss
            if self.config['use_lars']:
                total_loss = base_loss
            else:
                total_loss = base_loss + l2_loss
   
            total_loss = tf.identity(total_loss, name = 'total_loss')


            if mode == tf.estimator.ModeKeys.EVAL:
                with tf.device(None):
                    metrics = self.layers.get_accuracy( labels, predicted_classes, logits, self.config)

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            assert (mode == tf.estimator.ModeKeys.TRAIN)

            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #total_loss = tf.add_n([tf.saturate_cast(loss, self.config['dtype']) ] + reg_losses, name='total_loss')
            #total_loss = tf.add_n([loss], name='total_loss')
    
            batch_size = tf.shape(inputs)[0]
    
            global_step = tf.train.get_global_step()
            with tf.device('/cpu:0'):
                learning_rate = self.hyper_param.get_learning_rate()

            #-----------------------batchsize scaling----------------------------------
            momentum = self.config['momentum'][0]
            #------------------------------end------------------------------------------
 
            opt = tf.train.MomentumOptimizer(
                learning_rate, momentum, use_nesterov=self.config['use_nesterov'])
            opt=NPUDistributedOptimizer(opt) 
            if self.config['accelerator'] == 'gpu':
                opt = self.optimizer.get_lbs_optimizer(opt)           
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
            with tf.control_dependencies(update_ops):
                if self.config['accelerator'] == 'gpu':
                    gate_gradients = (tf.train.Optimizer.GATE_NONE)
                    grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)
                    train_op = opt.apply_gradients( grads_and_vars,global_step = global_step)
                else:
                    with tf.name_scope('loss_scale'):
                        loss_scale = float( self.config['loss_scale'] )
                        scaled_grads_and_vars = opt.compute_gradients( total_loss * loss_scale )
                        unscaled_grads_and_vars = [ (g/loss_scale, v)  for g,v in scaled_grads_and_vars ]


            #-----------------------------------------Lars------------------------------------------
                        with tf.name_scope('LARS'):
                            fp32_grads_and_vars = [ (tf.cast(g, tf.float32), v)  for g,v in unscaled_grads_and_vars ]
                            grad_var_list = []
                            
                            if self.config['use_lars']:
                                if self.config['accelerator'] == 'gpu':
                                    for g, var in  fp32_grads_and_vars: 
    
                                        if 'BatchNorm' not in var.name and 'bias' not in var.name:
                                            grad_norm = tf.norm(g,ord='euclidean') 
                                            weight_norm = tf.norm(var,ord='euclidean')
                                            grad_norm_wd = tf.add( grad_norm,  tf.multiply( self.config['weight_decay'] , weight_norm ) )
                                            rescale_factor = tf.div( tf.multiply(0.001, weight_norm), tf.add(grad_norm_wd, tf.constant(1e-5, tf.float32)) )
                                            decayed_g = tf.add( g, tf.multiply(self.config['weight_decay'], var ) )
    
                                            with tf.name_scope('lars_grad'):
                                                g = tf.multiply(rescale_factor, decayed_g)
    
                                        g_and_v = ( g, var )
                                        grad_var_list.append( g_and_v )
    
                                elif self.config['accelerator'] == '1980':
                                    print('lars9999999999999999999999')
                                    g_list_bn_bias = []
                                    var_list_bn_bias = []
                                    g_list_else = []
                                    var_list_else = []
                                    for g, var in fp32_grads_and_vars: 
                                        if 'BatchNorm' not in var.name and 'bias' not in var.name:
                                            g_list_else.append(g)
                                            var_list_else.append(var)
                                        else:
                                            g_list_bn_bias.append(g)
                                            var_list_bn_bias.append(var)
    
    
                                    g_list_else_lars = npu_ops.LARS(inputs_w=var_list_else, 
                                                    inputs_g=g_list_else, 
                                                    weight_decay=self.config['weight_decay'],
                                                    hyperpara=0.001,
                                                    epsilon=1e-5)
    
                                    g_list_lars = g_list_bn_bias + g_list_else_lars
                                    var_list = var_list_bn_bias + var_list_else
    
                                    for (g, var) in zip(g_list_lars,var_list):
                                        g_and_v = ( g, var )
                                        grad_var_list.append( g_and_v )
    
    
                            else:
                                print('do not use lars111111111111111111')
                                for g, var in  fp32_grads_and_vars:
                                    #if 'BatchNorm' not in var.name and 'bias' not in var.name:
                                    #    decayed_g = tf.add( g, tf.multiply( self.config['weight_decay'], var ) )
                                    #    g = decayed_g
                                    g_and_v = ( g, var )
                                    grad_var_list.append( g_and_v )
            #-----------------------------------------end Lars------------------------------------------




                        train_op = opt.apply_gradients( grad_var_list, global_step = global_step )

            train_op = tf.group(train_op)

            #with tf.device('/cpu:0'):
                #tf.summary.scalar('total_loss', total_loss)
                #tf.summary.scalar('base_loss', base_loss)
                #tf.summary.scalar('learning_rate', learning_rate)
                #tf.contrib.summary.flush()
#                if self.config['do_checkpoint']:
#                    summary_hook = tf.train.SummarySaverHook( save_steps=20, 
#                                                        output_dir=self.config['log_dir']+'/train_summary',
#                                                        summary_op = tf.summary.merge_all() ) 

            #return  tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=[summary_hook] )\
            #                   if self.config['do_checkpoint'] else tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op )
            return   tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op )
          
            # return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)



    def get_model_func(self): 
        model_name = self.config['model_name']
        if model_name.startswith('resnet'):
            nlayer = int(model_name[len('resnet'):])
            return lambda images, *args, **kwargs: \
                resnet.inference_resnet_v1(self.config,images, nlayer, *args, **kwargs)
        else:
            raise ValueError("Invalid model type: %s" % model_name)







        



