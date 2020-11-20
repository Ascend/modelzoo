import tensorflow as tf
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers

def inference_alexnet_impl(inputs, is_training=True):

    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4), padding='valid',use_bias=True,activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True)
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True)
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True)

    return x


# def inference_alexnet_impl_he_uniform(inputs, is_training=True):
#
#     x = inputs
#     # conv11*11
#     x = tf.pad(x, paddings=[[0,0],[2,1],[2,1],[0,0]], mode="CONSTANT")
#     x = conv_layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4), padding='valid',use_bias=True,activation=tf.nn.relu, kernel_initializer=tf.initializers.he_uniform(5))
#     x = pooling_layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')
#
#     # conv5*5
#     x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
#     x = conv_layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
#     x = pooling_layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')
#
#     # conv3*3
#     x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
#     x = conv_layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
#
#     x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
#     x = conv_layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
#
#     x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
#     x = conv_layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
#
#     x = pooling_layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')
#
#     x = tf.reshape(x, [-1, 256*6*6])
#
#     # fc layers
#     x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
#     x = tf.layers.dropout(x, 0.65, training=is_training)
#
#     x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
#     x = tf.layers.dropout(x, 0.65, training=is_training)
#
#     x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
#
#     return x

#
def inference_alexnet_impl_he_uniform(inputs, is_training=True):

    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,1],[2,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4), padding='valid',use_bias=True,activation=tf.nn.relu, kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))

    return x

def inference_alexnet_impl_he_uniform_custom(inputs, is_training=True):
    '''
      to be consistent with ME  default weight initialization

    '''

    scale =1.0/3.0 
    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4),
                         padding='valid',use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.dropout(x, 0.65, training=is_training)

    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    return x




def inference(config, inputs,training=False):
    """Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    if config['alexnet_version'] == 'he_uniform':
        return inference_alexnet_impl_he_uniform(inputs,is_training= training)
    elif config['alexnet_version'] == 'xavier':
        return inference_alexnet_impl(inputs,is_training= training)
    elif config['alexnet_version'] == 'he_uniform_custom':
        return inference_alexnet_impl_he_uniform_custom(inputs,is_training= training)
    else:
        raise ValueError("Invalid resnet version")
