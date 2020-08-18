import tensorflow as tf

from npu_bridge.estimator import npu_ops

# vgg with initialization method in gluoncv
def vgg_impl(inputs, is_training=True):
    x = inputs

    # conv1
    x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp1
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # covn2
    x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp2
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv3
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp3
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv4
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp4
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv5
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp5
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    x = tf.reshape(x, [-1, 7 * 7 * 512])

    # fc6
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    # drop6
    if is_training:
        x = npu_ops.dropout(x, 0.5)
    # fc7
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    # drop7
    if is_training:
        x = npu_ops.dropout(x, 0.5)
    # fc8
    x = tf.layers.dense(x, 1000, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    return x

