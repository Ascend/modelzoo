import tensorflow as tf
import layers as lay

nn = lay.Layers()


def se_block(x, ch_out, ratio=2, is_training=True):
    x = nn.conv2d(x, ch_out)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.conv2d(x, ch_out)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    se = nn.global_avgpool(x)
    out_shape = se.shape[-1]
    se = nn.dense(se, out_shape // ratio, use_bias=False)
    se = nn.relu(se)
    se = nn.dense(se, out_shape, use_bias=False)
    se = nn.sigmoid(se)
    x = tf.multiply(x, se)
    return x


def conv_block_a(x, ch_out, is_training):
    x = nn.conv2d(x, ch_out)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.conv2d(x, ch_out)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.max_pool(x)
    return x


def conv_block_b(x, ch_out, is_training):
    x = nn.conv2d(x, ch_out, use_bias=True)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.conv2d(x, ch_out, use_bias=True)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.conv2d(x, ch_out, use_bias=True)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.conv2d(x, ch_out, use_bias=True)
    x = nn.batch_norm(x, is_training)
    x = nn.relu(x)
    x = nn.max_pool(x)
    return x


def encoder1(x, is_training):
    x1 = conv_block_a(x, ch_out=64, is_training=is_training)
    x2 = conv_block_a(x1, ch_out=128, is_training=is_training)
    x3 = conv_block_b(x2, ch_out=256, is_training=is_training)
    x4 = conv_block_b(x3, ch_out=512, is_training=is_training)
    x5 = conv_block_b(x4, ch_out=512, is_training=is_training)
    return x5, [x1, x2, x3, x4]


def decoder1(inputs, skip_connections, is_training):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = nn.upsample(x)
        x = tf.concat([x, skip_connections[i]], axis=-1)
        x = se_block(x, f, is_training=is_training)

    return x


def encoder2(inputs, is_training):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = se_block(x, f, is_training=is_training)
        skip_connections.append(x)
        x = nn.max_pool(x)

    return x, skip_connections


def decoder2(x, skip_1, skip_2, is_training):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()

    for i, f in enumerate(num_filters):
        x = nn.upsample(x)
        x = tf.concat([x, skip_1[i], skip_2[i]], axis=-1)
        x = se_block(x, f, is_training=is_training)

    return x


def output_block(x):
    x = nn.conv2d(x, ch_out=1, kernel_size=1)
    x = nn.sigmoid(x)
    return x


def aspp(x, is_training):
    y1 = nn.global_avgpool(x)
    y1 = nn.conv2d(y1, ch_out=64, kernel_size=1)
    y1 = nn.batch_norm(y1, is_training=is_training)
    y1 = nn.relu(y1)
    y1 = nn.upsample(y1, ratio=(8, 10))

    y2 = nn.conv2d(x, ch_out=64, kernel_size=1, use_bias=False)
    y2 = nn.batch_norm(y2, is_training=is_training)
    y2 = nn.relu(y2)

    y3 = nn.conv2d(x, ch_out=64, kernel_size=3, rate=6, use_bias=False)
    y3 = nn.batch_norm(y3, is_training=is_training)
    y3 = nn.relu(y3)

    y4 = nn.conv2d(x, ch_out=64, kernel_size=3, rate=12, use_bias=False)
    y4 = nn.batch_norm(y4, is_training=is_training)
    y4 = nn.relu(y4)

    y5 = nn.conv2d(x, ch_out=64, kernel_size=3, rate=18, use_bias=False)
    y5 = nn.batch_norm(y5, is_training=is_training)
    y5 = nn.relu(y5)

    y5 = tf.concat([y1, y2, y3, y4, y5], axis=-1)

    y5 = nn.conv2d(y5, ch_out=64, kernel_size=1, use_bias=False)
    y5 = nn.batch_norm(y5, is_training=is_training)
    y5 = nn.relu(y5)

    return y5


def doubleunet(x, is_training=True):
    inputs = x
    x, skip_1 = encoder1(x, is_training=is_training)
    x = aspp(x, is_training=is_training)
    x = decoder1(x, skip_1, is_training=is_training)
    outputs1 = output_block(x)
    inputs = nn.max_pool(inputs)
    x = inputs * outputs1

    x, skip_2 = encoder2(x, is_training=is_training)
    x = aspp(x, is_training=is_training)
    x = decoder2(x, skip_1, skip_2, is_training=is_training)
    outputs2 = output_block(x)

    outputs = tf.concat([outputs1, outputs2], axis=-1)
    outputs = nn.conv2d(outputs, 2, stride=1)
    outputs = nn.batch_norm(outputs, is_training=is_training)
    outputs = nn.relu(outputs)
    outputs = nn.upsample(outputs, name='output')
    return outputs
