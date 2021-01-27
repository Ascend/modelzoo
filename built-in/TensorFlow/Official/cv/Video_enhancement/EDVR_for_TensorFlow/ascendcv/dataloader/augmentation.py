import os
import numpy as np
import tensorflow as tf


class AugmentNoise(object):
    def __init__(self, **kwargs):
        self.noise_type = kwargs.get('noise_type', 'clean')
        self._random_seed = kwargs.get('random_seed', None)
        self.random_seed = self._random_seed
        self.params = kwargs

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

    def apply(self, clean_data):
        if isinstance(clean_data, tf.Tensor):
            return self.apply_tf(clean_data)
        elif isinstance(clean_data, np.ndarray):
            return self.apply_numpy(clean_data)
        else:
            raise TypeError(f'Unrecognized data type of `clean_data` with {type(clean_data)}')

    def apply_numpy(self, clean_data):
        shape = clean_data.shape

        if self.noise_type == 'gaussian':
            mean = self.params.get('mean', 0.)
            std = self.params.get('std', 0.2)
            noise = std * np.random.randn(shape) + mean
            noisy = np.clip(clean_data + noise, 0., 1.)
        elif self.noise_type == 'salt-pepper':
            noisy = np.array(clean_data)
            amount = self.params.get('amount', 0.001)
            s_vs_p = self.params.get('s_vs_p', 0.5)
            num_salt = np.ceil(amount * np.prod(shape) * s_vs_p)
            coord = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
            noisy[coord] = 1.

            num_pepper = np.ceil(amount * np.prod(shape) * (1. - s_vs_p))
            coord = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
            noisy[coord] = 0.
        elif self.noise_type == 'poisson':
            lam = self.params.get('lambda', 255)
            vals = 2 ** np.ceil(np.log2(lam))
            noisy = np.random.poisson(clean_data * vals) / float(vals)
        elif self.noise_type == 'speckle':
            gauss = np.random.randn(shape)
            noisy = clean_data + clean_data * gauss
            noisy = np.clip(noisy, 0., 1.)
        elif self.noise_type == 'clean':
            noisy = clean_data
        else:
            raise ValueError(f'Unrecognized noise type `{self.noise_type}`')

        return noisy

    def apply_tf(self, clean_data):
        shape = clean_data.get_shape().as_list()

        if self.noise_type == 'gaussian':
            mean = self.params.get('mean', 0.)
            std = self.params.get('std', 0.2)
            noise = tf.random.normal(shape, mean=mean, stddev=std)
            noisy = tf.clip_by_value(clean_data + noise, 0., 1.)
        elif self.noise_type == 'salt-pepper':
            noisy = tf.identity(clean_data)
            ones = np.ones(shape).astype(np.float32)
            amount = self.params.get('amount', 0.001)
            s_vs_p = self.params.get('s_vs_p', 0.5)
            num_salt = np.ceil(amount * np.prod(shape) * s_vs_p)
            coord = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
            ones[coord] = 0.
            ones = tf.convert_to_tensor(ones)
            noisy = noisy * ones + 1. - ones

            ones = np.ones(shape).astype(np.float32)
            num_pepper = np.ceil(amount * np.prod(shape) * (1. - s_vs_p))
            coord = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
            ones[coord] = 0.
            ones = tf.convert_to_tensor(ones)
            noisy = noisy * ones
        elif self.noise_type == 'poisson':
            lam = self.params.get('lambda', 10)
            vals = 2 ** np.ceil(np.log2(lam))
            noisy = tf.random.poisson(clean_data * vals.astype(np.float32), shape=shape)
        elif self.noise_type == 'speckle':
            gauss = tf.random.normal(shape)
            noisy = clean_data + clean_data * gauss
            noisy = tf.clip_by_value(noisy, 0., 1.)
        elif self.noise_type == 'clean':
            noisy = tf.identity(clean_data)
        else:
            raise ValueError(f'Unrecognized noise type `{self.noise_type}`')

        return noisy
