# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         zx-generator
# Description:
# Author:       zx
# Date:         2020/9/29
# -------------------------------------------------------------------------------
import time

import numpy as np

from config.config import Config, Configurable
from data.generator_enqueuer import GeneratorEnqueuer


def get_train_data_loader(yaml):
    conf = Config()
    # compile yaml to get class args
    experiment_args = conf.compile(conf.load(yaml))['train_data']
    # get class instance from yaml args
    # print(json.dumps(experiment_args))
    train_data_loader = Configurable.construct_class_from_config(experiment_args)
    return train_data_loader, experiment_args


def generator(yaml="../experiments/base_totaltext.yaml", is_training=True, batch_size=8):
    import os
    yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml)
    train_data_loader, experiment_args = get_train_data_loader(yaml)
    image_size = experiment_args["processes"][1]["size"][0]

    # get train data length and init train indices
    dataset_size = train_data_loader.__len__()
    indices = np.arange(dataset_size)
    if is_training:
        np.random.shuffle(indices)
    current_idx = 0
    while True:
        batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32)
        batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
        batch_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
        batch_thresh_maps = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
        batch_thresh_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
        for i in range(batch_size):
            image_dict, image_label = train_data_loader.__getitem__(indices[current_idx])
            batch_images[i] = image_dict["image"]
            batch_gts[i] = image_dict["gt"]
            batch_masks[i] = image_dict["mask"]
            batch_thresh_maps[i] = image_dict["thresh_map"]
            batch_thresh_masks[i] = image_dict["thresh_mask"]
            current_idx += 1
            if current_idx >= dataset_size:
                if is_training:
                    np.random.shuffle(indices)
                current_idx = 0
        yield [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]


def get_batch(num_workers):
    try:
        # print("get_batch")
        # window system need to change use_multiprocessing to False
        enqueuer = GeneratorEnqueuer(generator(), use_multiprocessing=True)
        # print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
