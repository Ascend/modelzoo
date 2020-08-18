import tensorflow as tf
#from tensorflow.contrib.hccl.python.ops import hccl_ops
from npu_bridge.hccl import hccl_ops

class Layers:
 
    def get_accuracy(self, labels, predicted_classes, logits, config):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes) 
        top5acc = tf.metrics.mean(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
        if config['rank_size'] == 1:
            newaccuracy = (accuracy[0], accuracy[1])
            newtop5acc = (top5acc[0], top5acc[1])
        else:
            newaccuracy = (hccl_ops.allreduce(accuracy[0],"sum")/config['rank_size'], accuracy[1])
            newtop5acc = (hccl_ops.allreduce(top5acc[0],"sum")/config['rank_size'], top5acc[1])
        metrics = {'val-top1acc': newaccuracy, 'val-top5acc': newtop5acc}
        return metrics




