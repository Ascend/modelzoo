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
import cv2
import os
import zipfile


######################## color augmentation will be processed during runtime! #############################
######################## look 'Dataloader.color_augment()' in 'data_loader.py' ############################
class DataPreprocessor:
    def __init__(self):
        output_path = 'data/data_preprocessed'
        input_path = 'data/dog_and_cat_25000_split'

        self.output_dir_train = os.path.join(output_path, 'train')
        self.input_dir_train = os.path.join(input_path, 'train')

        #if not(os.path.exists(input_path)):
        #    print("[unzip data]")
        #    os.mkdir(input_path)
        #    train_zip = zipfile.ZipFile('C:/kaggle/input/dogs-vs-cats/train.zip')
        #    test_zip = zipfile.ZipFile('C:/kaggle/input/dogs-vs-cats/test1.zip')
        #    train_zip.extractall(input_path)
        #    test_zip.extractall(input_path)

        if not (os.path.exists(output_path)):
            os.mkdir(output_path)
            os.mkdir(self.output_dir_train)

        self.file_list_train = os.listdir(self.input_dir_train)


    def resize(self, img):
        h = img.shape[0]
        w = img.shape[1]

        if h > w:
            return cv2.resize(img, (256, int(h / w * 256)), interpolation = cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, (int(w / h * 256), 256), interpolation=cv2.INTER_LINEAR)


    def crop_flip(self, img):
        h = img.shape[0]
        w = img.shape[1]

        ####left_top
        lt = img[:227, :227, :]
        ####right_top
        rt = img[:227, w - 227:, :]
        ####left_bottom
        lb = img[h - 227:, :227, :]
        ####right_bottom
        rb = img[h - 227:, w - 227:, :]
        ####center
        c = img[h//2 - 113:h//2 + 114, w//2 - 113:w//2 + 114, :]

        return [lt, cv2.flip(lt, 1), rt, cv2.flip(rt, 1), lb, cv2.flip(lb, 1), rb, cv2.flip(rb, 1), c, cv2.flip(c, 1)]


    def process_and_save(self, dir_input, dir_output, filenames):
        for filename in filenames:
            #print("file name=" + filename)
            path = os.path.join(dir_input, filename)
            img = cv2.imread(path)
            img_list = self.crop_flip(self.resize(img))

            file_num = eval(filename.split('.')[-2])
            file_class = filename.split('.')[-3]

            for i in range(len(img_list)):
                cv2.imwrite(os.path.join(dir_output, '%s.%09d%1d.jpg' % (file_class, file_num, i)), img_list[i])

    def run(self):
        self.process_and_save(self.input_dir_train, self.output_dir_train, self.file_list_train)


# if __name__ == "__main__":
#     DP = DataPreprocessor()
#     DP.run()







