import os
import tensorflow as tf
import numpy as np

from ascendcv.dataloader.minibatch import Minibatch, TestMinibatch, DataLoader_tfTest, DataLoader_tensorslice
from ascendcv.dataloader.dataloader import PrefetchGenerator
from ascendcv.dataloader.noise import get_noise_augmenter
from ascendvsr.config.utils import convert_to_dict


def build_train_dataloader(read_mode, batch_size, scale, set_file, num_frames, in_size, data_config):
    noise_options = convert_to_dict(data_config.noise, [])
    noise_augmenter = get_noise_augmenter(**noise_options)

    if read_mode == 'python':
        minibatch = Minibatch(
            data_dir=data_config.data_dir,
            set_file=set_file,
            batch_size=batch_size,
            num_frames=num_frames,
            scale=scale,
            in_size=in_size,
            noise_augmenter=noise_augmenter
        )
        loader = PrefetchGenerator(minibatch, data_config.num_threads, data_config.train_data_queue_size)
        return loader
    elif read_mode == 'tf':
        minidata = DataLoader_tensorslice(
            data_dir=data_config.data_dir,
            set_file=set_file,
            batch_size=batch_size,
            num_frames=num_frames,
            scale=scale,
            in_size=in_size,
            noise_augmenter=noise_augmenter)
        return minidata.batch_list
    else:
        raise ValueError


def build_test_dataloader(batch_size, scale, set_file, num_frames, data_config):
    dataloader = TestMinibatch(
        data_dir=data_config.data_dir,
        set_file=set_file,
        batch_size=batch_size,
        num_frames=num_frames,
        scale=scale
    )
    return dataloader
