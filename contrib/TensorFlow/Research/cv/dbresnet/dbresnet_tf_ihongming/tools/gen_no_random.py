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

from config.config import Config, Configurable
from data.generator_enqueuer import GeneratorEnqueuer


def get_train_data_loader(yaml):
    conf = Config()
    experiment_args = conf.compile(conf.load(yaml))['train_data']
    train_data_loader = Configurable.construct_class_from_config(experiment_args)
    return train_data_loader, experiment_args


def generator(yaml="./experiments/base_totaltext.yaml", is_training=True, batch_size=8):
    print(yaml)
    import os
    yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml)
    print(yaml)
    print(os.path.dirname(os.path.abspath(__file__)))
    print(">>>>>>>>")
    train_data_loader, experiment_args = get_train_data_loader(yaml)
    image_size = experiment_args["processes"][1]["size"][0]
    # image_dict, image_label = train_data_loader.__getitem__(1)
    # print(image_dict.keys())
    # print(image_label)
    # print(image_size)


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


if __name__ == '__main__':
    import tensorflow as tf
    from tqdm import tqdm
    from cv2 import cv2
    train_gen = generator()
    writer = tf.python_io.TFRecordWriter('%s.tfrecord' % 'no_random')

    for i in tqdm(range(2)):
        train_data = next(train_gen)
        cv2.imwrite("temps/"+str(i)+".jpeg", train_data[0][0])
        # cv2.imshow("hello", train_data[0][0])
        # cv2.waitKey(0)
        
        

        features = {}
        features['input_images'] = tf.train.Feature(float_list = tf.train.FloatList(value=train_data[0].reshape(-1)))
        features['input_score_maps'] = tf.train.Feature(float_list = tf.train.FloatList(value=train_data[1].reshape(-1)))
        features['input_score_masks'] = tf.train.Feature(float_list = tf.train.FloatList(value=train_data[2].reshape(-1)))
        features['input_threshold_maps'] = tf.train.Feature(float_list = tf.train.FloatList(value=train_data[3].reshape(-1)))
        features['input_threshold_masks'] = tf.train.Feature(float_list = tf.train.FloatList(value=train_data[4].reshape(-1)))
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)

    writer.close()


