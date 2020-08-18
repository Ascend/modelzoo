import tensorflow as tf

class Loss:
    def __init__(self,config):
        self.config = config 

    def get_loss(self, logits, labels):
        labels_one_hot = tf.one_hot(labels, self.config['num_classes'])
        loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels_one_hot,label_smoothing=self.config['label_smoothing'])
        loss = tf.identity(loss, name='loss')
        return loss

    def get_total_loss(self, loss):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + reg_losses, name='total_loss')
        return total_loss
 

    def optimize_loss(self, total_loss, opt):
        gate_gradients = (tf.train.Optimizer.GATE_NONE)
        # grads_and_vars = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True, gate_gradients=gate_gradients)
        grads_and_vars = opt.compute_gradients(total_loss, gate_gradients=gate_gradients)

        # train_op = opt.apply_gradients( grads_and_vars, global_step=None )
        train_op = opt.apply_gradients( grads_and_vars)

        return train_op

   


        



