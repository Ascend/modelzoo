import tensorflow as tf


def depth_to_space(x, scale, use_default=False):
    # Ascend implementation of tf.depth_to_space is not accurate so far
    # Thanks to Huang Wei h00573990
    if use_default:
        out = tf.depth_to_space(x, scale)
    else:
        b, h, w, c = list(map(int, x.shape))
        out = tf.reshape(x, [b, h, w, scale, scale, -1])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [b, h * scale, w * scale, -1])
    return out


def resize(x, size, align_corners=False, name=None, half_pixel_centers=False, method='bicubic'):
    if method == 'bicubic':
        upsampling = tf.image.resize_bicubic
    elif method == 'bilinear':
        upsampling = tf.image.resize_bilinear
    else:
        raise ValueError
    return upsampling(x, size=size, align_corners=align_corners, name=name, half_pixel_centers=half_pixel_centers)

