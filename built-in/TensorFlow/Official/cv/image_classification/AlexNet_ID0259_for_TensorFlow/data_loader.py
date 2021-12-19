#
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
import numpy as np
import os
import cv2


######################## color augmentation is implemented in this module (to be processed during runtime) #############
######################## look 'Dataloader.color_augment()'    ##########################################################
class DataLoader:
    def __init__(self, validation_len=1000,train_dir = "",mul_rank_size=1, mul_device_id=0):
        self.train_dir = train_dir
        self.mul_rank_size = mul_rank_size
        self.mul_device_id = mul_device_id
        #test_dir = 'C:/kaggle/input/data_raw/test1'

        self.train_path_list = [os.path.join(self.train_dir, x) for x in os.listdir(self.train_dir)] # 250000
        if mul_rank_size != 1:
            len_single = int(len(self.train_path_list) / mul_rank_size)
            self.train_path_list = self.train_path_list[mul_device_id*len_single:(mul_device_id+1)*len_single]
        np.random.shuffle(self.train_path_list)

        self.val_path_list = self.train_path_list[-validation_len:] # 10000
        self.train_path_list = self.train_path_list[:-validation_len] # 240000
        #self.test_path_list = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

        self.idx_train = [i for i in range(len(self.train_path_list))]
        np.random.shuffle(self.idx_train)
        self.idx_val = [i for i in range(validation_len)]
        #self.idx_test = range(len(self.test_path_list))
        #np.random.shuffle(self.idx_train)

        self.cursor_train = 0
        self.cursor_val = 0
        #self.cursor_test = 0

        self.class_dict = {'cat': 0, 'dog': 1}

    def color_augment(self, img):
        img_reshaped = img.reshape(227 * 227, 3).astype(np.float32)
        mean = np.mean(img_reshaped, axis=0)
        std = np.std(img_reshaped, axis=0)
        img_norm = (img_reshaped - mean) / std

        cov = np.cov(img_norm, rowvar=False)
        lambdas, p = np.linalg.eig(cov)
        alphas = np.random.normal(0, 0.1, 3)
        delta = np.dot(p, alphas * lambdas)

        img_result = img_norm + delta
        img_result = img_result * std + mean
        img_result = np.maximum(np.minimum(img_result, 255), 0).reshape(227, 227, 3)
        return img_result

    def next_train(self, batch_size, lock):
        batch_img = np.empty((
            batch_size,
            227,
            227,
            3
        ), dtype=np.float32)

        batch_label = np.zeros((
            batch_size,
            2
        ), dtype=np.float32) #one-hot encoding

        for idx, val in enumerate(self.idx_train[self.cursor_train:self.cursor_train + batch_size]):
            pathname = self.train_path_list[val]
            class_idx = self.class_dict[pathname.split('.')[-3][-3:]]

            batch_img[idx] = self.color_augment(cv2.imread(pathname))
            batch_label[idx, class_idx] = 1

        lock.acquire()
        self.cursor_train += batch_size
        if self.cursor_train + batch_size > len(self.idx_train):
            self.cursor_train = 0
            np.random.shuffle(self.idx_train)
        lock.release()
        
        return batch_img, batch_label

    def next_val(self, batch_size):
        batch_img = np.empty((
            batch_size,
            227,
            227,
            3
        ), dtype=np.float32)

        batch_label = np.zeros((
            batch_size,
            2
        ), dtype=np.float32)  # one-hot encoding

        for idx, val in enumerate(self.idx_val[self.cursor_val:self.cursor_val + batch_size]):
            pathname = self.val_path_list[val]
            class_idx = self.class_dict[pathname.split('.')[-3][-3:]]

            batch_img[idx] = cv2.imread(pathname)
            batch_label[idx, class_idx] = 1

        self.cursor_val += batch_size
        if self.cursor_val + batch_size > len(self.idx_val):
            self.cursor_val = 0

        return batch_img, batch_label


# if __name__ == "__main__":
#     DL = DataLoader()
#     img, lb = DL.next_train(100)
#     print()
