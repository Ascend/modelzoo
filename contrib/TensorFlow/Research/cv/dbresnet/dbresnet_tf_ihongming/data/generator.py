# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import numpy as np
import sys
import os
import pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))

from config.config import Config, Configurable
from data.generator_enqueuer import GeneratorEnqueuer


def get_train_data_loader(yaml):
    conf = Config()
    experiment_args = conf.compile(conf.load(yaml))['train_data']
    train_data_loader = Configurable.construct_class_from_config(experiment_args)
    return train_data_loader, experiment_args

def get_val_data_loader(yaml):
    conf = Config()
    experiment_args = conf.compile(conf.load(yaml))['validate_data']
    val_data_loader = Configurable.construct_class_from_config(experiment_args)
    return val_data_loader, experiment_args

def pad_list(batch_gts, batch_masks):
    positive_mask = (batch_gts * batch_masks)
    positive_count = np.sum(positive_mask)
    max_length = batch_gts.size

    ones_mask = np.ones((min(max_length, positive_count*3).astype(int),), dtype=np.float32)
    batch_neg_mask = np.pad(ones_mask, (0, max_length - ones_mask.size))

    return batch_neg_mask

def generator(yaml="../config/base_totaltext.yaml", is_training=True, batch_size=8):
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
        batch_topk_mask = np.zeros([batch_size*image_size*image_size], dtype=np.float32)
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
        # batch_positive = np.sum(batch_gts).astype(int)
        batch_topk_mask = pad_list(batch_gts, batch_masks)
        yield [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks, batch_topk_mask]


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


if __name__ == '__main__':
    train_gen = generator()
    train_data = next(train_gen)
    print(train_data[5])
    print(np.sum(train_data[1]*train_data[2]))
    print(np.sum(train_data[5]))