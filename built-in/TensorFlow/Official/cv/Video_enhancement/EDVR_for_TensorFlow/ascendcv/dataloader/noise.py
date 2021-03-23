import os
import numpy as np
import tensorflow as tf


class NoiseAugmentation(object):
    def __init__(self, **kwargs):
        self.random_seed = kwargs.get('random_seed', None)

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
        raise NotImplementedError

    def apply_tf(self, clean_data):
        raise NotImplementedError


class GaussianNoise(NoiseAugmentation):
    def __init__(self, mean=0., std=0.05, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def apply_numpy(self, clean_data):
        shape = clean_data.shape
        noise = self.std * np.random.randn(*shape) + self.mean
        noisy = np.clip(clean_data + noise, 0., 1.)
        return noisy

    def apply_tf(self, clean_data):
        shape = clean_data.get_shape().as_list()
        noise = tf.random.normal(shape, mean=self.mean, stddev=self.std)
        noisy = tf.clip_by_value(clean_data + noise, 0., 1.)
        return noisy


class SaltPepperNoise(NoiseAugmentation):
    def __init__(self, amount=0.005, salt_ratio=0.5, **kwargs):
        super(SaltPepperNoise, self).__init__(**kwargs)
        self.amount = amount
        self.salt_noise_ratio = salt_ratio

    def apply_numpy(self, clean_data):
        shape = clean_data.shape
        noisy = np.array(clean_data)

        num_salt = np.ceil(self.amount * np.prod(shape) * self.salt_noise_ratio)
        coord = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
        noisy[coord] = 1.

        num_pepper = np.ceil(self.amount * np.prod(shape) * (1. - self.salt_noise_ratio))
        coord = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
        noisy[coord] = 0.
        return noisy

    def apply_tf(self, clean_data):
        shape = clean_data.get_shape().as_list()

        noisy = tf.identity(clean_data)
        ones = np.ones(shape).astype(np.float32)
        num_salt = np.ceil(self.amount * np.prod(shape) * self.salt_noise_ratio)
        coord = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
        ones[coord] = 0.
        ones = tf.convert_to_tensor(ones)
        noisy = noisy * ones + 1. - ones

        ones = np.ones(shape).astype(np.float32)
        num_pepper = np.ceil(self.amount * np.prod(shape) * (1. - self.salt_noise_ratio))
        coord = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
        ones[coord] = 0.
        ones = tf.convert_to_tensor(ones)
        noisy = noisy * ones
        return noisy


class SpeckleNoise(NoiseAugmentation):
    def __init__(self, **kwargs):
        super(SpeckleNoise, self).__init__(**kwargs)

    def apply_numpy(self, clean_data):
        shape = clean_data.shape
        gauss = np.random.randn(*shape)
        noisy = clean_data + clean_data * gauss
        noisy = np.clip(noisy, 0., 1.)
        return noisy

    def apply_tf(self, clean_data):
        shape = clean_data.get_shape().as_list()
        gauss = tf.random.normal(shape)
        noisy = clean_data + clean_data * gauss
        noisy = tf.clip_by_value(noisy, 0., 1.)
        return noisy


class GaussianProcessNoise(NoiseAugmentation):
    def __init__(self, mean=0., min_std=0.01, max_std=0.1, **kwargs):
        super(GaussianProcessNoise, self).__init__(**kwargs)
        self.mean = mean
        self.min_std = min_std
        self.max_std = max_std

    def apply_numpy(self, clean_data):
        shape = clean_data.shape
        std = self.min_std + (self.max_std - self.min_std) * np.random.rand(*shape)
        noise = std * np.random.randn(*shape) + self.mean
        noisy = np.clip(clean_data + noise, 0., 1.)
        return noisy

    def apply_tf(self, clean_data):
        shape = clean_data.get_shape().as_list()
        std = tf.random.uniform(shape, minval=self.min_std, maxval=self.max_std)
        noise = tf.random.normal(shape, mean=0., stddev=1.) * std + self.mean
        noisy = tf.clip_by_value(clean_data + noise, 0., 1.)
        return noisy


class NoNoise(NoiseAugmentation):
    def __init__(self, **kwargs):
        super(NoNoise, self).__init__(**kwargs)

    def apply_numpy(self, clean_data):
        return clean_data

    def apply_tf(self, clean_data):
        return clean_data


class NoiseSequential(NoiseAugmentation):
    def __init__(self, list_of_noise_aug, **kwargs):
        super(NoiseSequential, self).__init__(**kwargs)
        self.noise_components = list_of_noise_aug

    def apply_numpy(self, clean_data):
        for noise_augmenter in self.noise_components:
            clean_data = noise_augmenter.apply_numpy(clean_data)
        return clean_data

    def apply_tf(self, clean_data):
        for noise_augmenter in self.noise_components:
            clean_data = noise_augmenter.apply_tf(clean_data)
        return clean_data


NOISE_AUGMENTER_MAP = {
    'clean': NoNoise,
    'gaussian': GaussianNoise,
    'gaussian process': GaussianProcessNoise,
    'speckle': SpeckleNoise,
    'salt-pepper': SaltPepperNoise
}


def get_noise_augmenter(**kwargs):
    noise_type = kwargs.get('noise_type', 'clean')
    noise_augmenter = NOISE_AUGMENTER_MAP[noise_type]
    return noise_augmenter(**kwargs)
