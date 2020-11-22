import tensorflow as tf

class Layers:
 
    def get_accuracy(self, labels, predicted_classes, logits, config):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes) 
        top5acc = tf.metrics.mean(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))

        newaccuracy = (accuracy[0], accuracy[1])
        newtop5acc = (top5acc[0], top5acc[1])

        metrics = {'val-top1acc': newaccuracy, 'val-top5acc': newtop5acc}
        return metrics




