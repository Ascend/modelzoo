
import tensorflow as tf
from . import alexnet, alexnet_helper
from trainers.train_helper import stage
# add MultiGPU
import horovod.tensorflow as hvd

_NUM_EXAMPLES_NAME="num_examples"


class Model(object):
    # def __init__(self, config, data, hyper_param, layers, optimizer, loss, logger):
    def __init__(self, config, data, hyper_param, layers):
        self.config = config
        self.data = data
        self.hyper_param = hyper_param
        self.layers = layers



    def get_estimator_model_func(self, features, labels, mode, params=None):
        labels = tf.reshape(labels, (-1,))  # Squash unnecessary unary dim
    
        model_func = self.get_model_func()
        inputs = features  
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.device('/gpu:0'):
            inputs = tf.cast(inputs, self.config['dtype'])
            with alexnet_helper.custom_getter_with_fp16_and_weight_decay(dtype=self.config['dtype'], weight_decay=self.config['weight_decay']):
                top_layer = model_func(inputs,  training=is_training)

            logits = top_layer
            predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
            logits = tf.cast(logits, tf.float32)

            labels_one_hot = tf.one_hot(labels, depth=1000)
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels_one_hot, label_smoothing=self.config['label_smoothing'])

            base_loss = tf.identity(loss, name='loss')  #

            # Add weight decay to the loss.
            # loss is computed using fp32 for numerical stability.
            l2_loss = self.config['weight_decay'] * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

            if self.config['use_lars']:
                total_loss = base_loss
            else:
                total_loss = base_loss + l2_loss
   
            total_loss = tf.identity(total_loss, name = 'total_loss')

            if mode == tf.estimator.ModeKeys.EVAL:
                with tf.device(None):
                    metrics = self.layers.get_accuracy( labels, predicted_classes, logits, self.config)

                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            assert (mode == tf.estimator.ModeKeys.TRAIN)
    
            global_step = tf.train.get_global_step()
            with tf.device('/cpu:0'):
                learning_rate = self.hyper_param.get_learning_rate()

            # optimizer config
            momentum = self.config['momentum'][0]
            opt = tf.train.MomentumOptimizer(
                learning_rate, momentum, use_nesterov=self.config['use_nesterov'])

            opt = hvd.DistributedOptimizer(opt)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []

            with tf.control_dependencies(update_ops):
                gate_gradients = (tf.train.Optimizer.GATE_NONE)
                grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

            train_op = tf.group(train_op)

            return   tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op )


    def get_model_func(self): 
        model_name = self.config['model_name']
        if model_name.startswith('alex'):
            return lambda images, * args, **kwargs: \
                alexnet.inference(self.config,images, *args, **kwargs)
        else:
            raise ValueError("Invalid model type: %s" % model_name)
