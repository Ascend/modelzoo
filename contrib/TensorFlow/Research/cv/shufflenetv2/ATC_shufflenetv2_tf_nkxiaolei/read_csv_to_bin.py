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
import os
import numpy as np
from keras.preprocessing.image import img_to_array

txt_path= "./ground_truth/public_test.txt"
file_path= "./input/"

with open(txt_path, 'r') as f:
     reader = f.readlines()
     i = 0
     for row in reader:
         if "PublicTest" in row:
             print("start to preprocess image file: %d"%i)
             data = row.split(",")[2].split(" ")
             data = list(map(int,data))
             img = np.array(data).reshape(48,48)
             img = img.astype(np.float) / 255.
             img = img_to_array(img)
             tmp = np.expand_dims(img, 0)
             tmp.tofile(file_path+"/"+str(i.__str__().zfill(6))+".bin")
             i+=1
