# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import tensorflow as tf
from tensorflow.python.util import nest
import os ,sys
import numpy as np
import common
from datasets import data_generator


class DataLoader:

    def __init__(self):
        #self.config = config

        
        self.num_training_samples = (data_generator
                                     ._DATASETS_INFORMATION[common.FLAGS.dataset]
                                     .splits_to_sizes[common.FLAGS.train_split])

        # num_evaluating_samples = get_num_records(self.eval_filenames)
        # self.config.num_training_samples = num_training_samples
        # self.config.num_evaluating_samples = 50000
        print( 'total num_training_sampels: %d' %  self.num_training_samples )
        print( 'common flags ', common.FLAGS.dataset)
        self.num_of_classes = {common.OUTPUT_TYPE: data_generator._DATASETS_INFORMATION[common.FLAGS.dataset].num_classes}

        #self.training_samples_per_rank = num_training_samples

    def get_train_input_fn(self):
        dataset = data_generator.Dataset(
            dataset_name=common.FLAGS.dataset,
            split_name=common.FLAGS.train_split,
            dataset_dir=common.FLAGS.dataset_dir,
            batch_size=common.FLAGS.train_batch_size,
            crop_size=[int(sz) for sz in common.FLAGS.train_crop_size],
            min_resize_value=common.FLAGS.min_resize_value,
            max_resize_value=common.FLAGS.max_resize_value,
            resize_factor=common.FLAGS.resize_factor,
            min_scale_factor=common.FLAGS.min_scale_factor,
            max_scale_factor=common.FLAGS.max_scale_factor,
            scale_factor_step_size=common.FLAGS.scale_factor_step_size,
            model_variant=common.FLAGS.model_variant,
            num_readers=4,
            is_training=True,
            should_shuffle=True,
            should_repeat=True)

        return dataset.get_iterator()

    def get_eval_input_fn(self):
        dataset = data_generator.Dataset(                                         
             dataset_name=common.FLAGS.dataset,
             split_name=common.FLAGS.eval_split,
             dataset_dir=common.FLAGS.dataset_dir,
             batch_size=common.FLAGS.eval_batch_size,
             crop_size=[int(sz) for sz in common.FLAGS.eval_crop_size],
             min_resize_value=common.FLAGS.min_resize_value,
             max_resize_value=common.FLAGS.max_resize_value,
             resize_factor=common.FLAGS.resize_factor,
             num_readers = 2,
      	     is_training = False,
	     should_shuffle = False,
             should_repeat = False)       
       
        return dataset.get_iterator()



