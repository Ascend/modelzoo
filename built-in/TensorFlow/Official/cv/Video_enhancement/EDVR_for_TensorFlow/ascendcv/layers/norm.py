import tensorflow as tf


def BatchNorm(x, momentum=0.999, is_train=True, name='BatchNorm'):
    output = tf.layers.batch_normalization(x, momentum=momentum, epsilon=1e-3, name=name, training=is_train)

    return output


class NormLayer(object):

    def __init__(self, cfg, is_train, name=None):
        super(NormLayer, self).__init__()
        self.type = cfg.get('type').lower()
        self.is_train = is_train
        self.name = name

    def _forward(self, x):
        if self.type == 'bn':
            return BatchNorm(x, is_train=self.is_train, name=self.name)
        else:
            raise NotImplementedError

    def __call__(self, x):
        shape = list(map(int, x.shape))
        if len(shape) == 5:
            # TODO
            # Ascend currently do not support 5D bn
            x_4d = tf.reshape(x, [-1] + shape[2:])
            x_4d = self._forward(x_4d)
            x = tf.reshape(x_4d, shape)
        else:
            x = self._forward(x)

        return x
