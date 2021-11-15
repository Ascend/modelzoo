"""
Misc
"""
import tensorflow as tf
import numpy as np
import random


def pair(x, dims=2):
    if isinstance(x, list) or isinstance(x, tuple):
        assert len(x) == dims
    elif isinstance(x, int):
        x = [x] * dims
    else:
        raise ValueError
    return x


def set_global_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
