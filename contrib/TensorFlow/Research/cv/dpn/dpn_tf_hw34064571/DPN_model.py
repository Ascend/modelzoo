
import tensorflow as tf
import tensorflow.contrib.slim as slim

def vgg_arg_scope(weight_decay=0.0005,
                  use_batch_norm=True,
                  batch_norm_decay=0.9997,
                  batch_norm_epsilon=0.001,
                  batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                  batch_norm_scale=False):
    """Defines the VGG arg scope.
    Args:
      weight_decay: The l2 regularization coefficient.
    Returns:
      An arg_scope.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': batch_norm_updates_collections,
        # use fused batch norm if possible.
        'fused': None,
        'scale': batch_norm_scale,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params) as arg_sc:
            return arg_sc

def output_block(x1, x2):
    a = tf.log(x1) - x2
    # b = tf.reduce_sum(a, axis=3, keep_dims=True)
    # out = tf.div(a, b)
    # out = tf.nn.softmax(out)
    return a

def bmin(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:
        axis = -1
    num_channels = shape[axis]
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_min(tf.reshape(inputs, shape), -2, keep_dims=False)
    return outputs

def local_conv2d(inputs, inputd, kernel_size, strides, output_shape):
    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_row, kernel_col = kernel_size[0], kernel_size[1]
    # inputs = tf.reduce_sum(inputs, axis=3, keep_dims=True)
    inputs = tf.pad(inputs, [[0, 0], [5, 4], [5, 4], [0, 0]], "CONSTANT")

    inputs = tf.extract_image_patches(inputs,
                                      [1, kernel_row, kernel_col, 1],
                                      [1, stride_row, stride_col, 1],
                                      [1, 1, 1, 1],
                                      padding='VALID')
    inputs = tf.reshape(inputs, [-1, output_row * output_col, kernel_col, kernel_row, 21])
    center_vector = inputs[:, :, 5:6, 5:6, :]
    vector_distance = tf.reduce_mean(inputs, 0, keep_dims=True) - tf.reduce_mean(center_vector, 0, keep_dims=True)
    vector_distance = tf.reduce_sum(vector_distance ** 2, axis=-1, keep_dims=True)
    weight1 = tf.Variable(0.5, trainable=True)
    weight2 = tf.Variable(0.5, trainable=True)
    vector_distance = weight1 * vector_distance + weight2 * inputd
    vector_distance = vector_distance + inputd
    vector_distance = vector_distance * center_vector
    inputs = tf.reduce_sum(inputs, axis=-1, keep_dims=True)
    inputs = inputs * vector_distance
    inputs = tf.reduce_sum(inputs, axis=[2, 3])
    inputs = tf.reshape(inputs, [-1, output_col, output_row, 21])
    return inputs

def dpn_model(inputs, inputd, is_training, _HEIGHT, _WIDTH, num_classes=21, scope='vgg_16'):
    print("-----------------------------start----------------------------------------")
    # pretrain encoder part
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        # vgg encoder part
        with tf.variable_scope(scope, 'vgg_16', [inputs]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # print("--------net1--------", net)
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # print("--------net2--------", net)
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # print("--------net3--------", net)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # print("--------net4-------", net)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')
            print("-----------------------------end----------------------------------------")
        # decoder part
        with tf.variable_scope('upsample'):
            net = tf.layers.conv2d(net, 512, 3, 1, padding="same")  # 1024s
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            # net = slim.conv2d(net, 1024, [7, 7], 1, padding="same")
            net = tf.layers.conv2d(net, 2048, 3, 1, padding="same")  # 1024s
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            # net = slim.conv2d(net, 1024, [1, 1], 1, padding="same")
            net = tf.layers.conv2d(net, 1024, 1, 1, padding="same")  # 1024s
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            output1 = slim.conv2d(net, num_classes, [1, 1], 1, padding="same")
            # output1 = tf.image.resize_images(net, [_HEIGHT, _WIDTH])
            output1 = tf.sigmoid(output1)

            # kernel = tf.Variable(tf.random.truncated_normal([320 * 320, 25 * 25 * 1, num_classes]),
            #                      trainable=True)  # 1 1
            print("-------------------------------local-------------------------------",output1)

            output2 = local_conv2d(output1, inputd, kernel_size=(10, 10), strides=(1, 1),
                                   output_shape=(64, 64))
            # print("-------------------------------local-------------------------------", output2)

            output2 = slim.conv2d(output2, 105, [9, 9], 1, padding='same', activation_fn=None)
            output2 = bmin(output2, num_units=5)
            out = output_block(output1, output2)
            out = tf.sigmoid(out)
            out = slim.conv2d(out, 21, [1, 1], 1, padding='same')
            # out = tf.image.resize_images(out, [_HEIGHT, _WIDTH])

            return out
