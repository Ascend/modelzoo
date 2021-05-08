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
import os
import sys
import shutil


def regroup_reds_dataset(target_dir, train_path, val_path, subset):
    """Regroup original REDS datasets.
    We merge train and validation data into one folder, and separate the
    validation clips in reds_dataset.py.
    There are 240 training clips (starting from 0 to 239),
    so we name the validation clip index starting from 240 to 269 (total 30
    validation clips).
    Args:
        target_dir (str): Source path to save the dataset, e.g. /data/reds/images
        train_path (str): Path to the train folder.
        val_path (str): Path to the validation folder.
        subset (str): Indicating the set name, 'truth' or 'blur4'
    """
    # move the validation data to the train folder
    tr_folders = os.listdir(train_path)
    for folder in tr_folders:
        source_folder = os.path.join(train_path, folder)
        output_folder = os.path.join(target_dir, f'{folder}', subset)
        shutil.move(source_folder, output_folder)

    val_folders = os.listdir(val_path)
    for folder in val_folders:
        new_folder_idx = int(folder) + 240
        source_folder = os.path.join(val_path, folder)
        output_folder = os.path.join(target_dir, f'{new_folder_idx:03d}', subset)
        shutil.move(source_folder, output_folder)

datadir = sys.argv[1]
target_dir = os.path.join(datadir, 'images')
os.makedirs(target_dir, exist_ok=True)

# train_sharp
train_path = os.path.join(datadir, 'train/train_sharp')
val_path = os.path.join(datadir, 'val/val_sharp')
regroup_reds_dataset(target_dir, train_path, val_path, 'truth')

# train_sharp_bicubic
train_path = os.path.join(datadir, 'train/train_sharp_bicubic/X4')
val_path = os.path.join(datadir, 'val/val_sharp_bicubic/X4')
regroup_reds_dataset(target_dir, train_path, val_path, 'blur4')
