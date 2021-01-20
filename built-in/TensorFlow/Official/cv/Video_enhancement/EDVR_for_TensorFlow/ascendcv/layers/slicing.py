import numbers
import tensorflow as tf


def tf_slicing(x, axis, slice_range, keep_dim=False):
    target_dim = int(x.shape[axis])
    ndim = len(x.shape)
    begin = list(ndim * [0])
    size = x.get_shape().as_list()

    if isinstance(slice_range, (list, tuple)):
        begin[axis] = slice_range[0]
        size[axis] = slice_range[1] - slice_range[0]
    elif isinstance(slice_range, numbers.Integral):
        begin[axis] = slice_range
        size[axis] = 1
    else:
        raise ValueError

    x_slice = tf.slice(x, begin, size)
    if size[axis] == 1 and not keep_dim:
        x_slice = tf.squeeze(x_slice, axis)

    return x_slice


def tf_split(x, num_or_size_splits, axis=0, num=None, keep_dims=False):
    x_list = tf.split(x, num_or_size_splits, axis, num)

    if not keep_dims:
        x_list2 = [tf.squeeze(x_, axis) for x_ in x_list]
        return x_list2

    return x_list

