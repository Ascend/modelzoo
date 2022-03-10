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

import datetime


############################
##        PREPARE         ##
############################
crops_p_img = 10    # Number of samples/crops taken per HR image (to get the target output size)
augment_img = True  # Augment data with flips (each image will generate an extra flipped image)


############################
##       SAVE/LOAD        ##
############################
load_model = True              # Should we load a saved model from memory, or create a new one?
save_dir   = 'save'            # Folder name where the model will be saved, relative to root
model_name = 'sobel_model.h5'  # Name of the model that is to be loaded and/or saved
# TODO: integrate those two feedbacks
model_json = 'model_architecture.json'
weights    = 'model_weights.h5'

def get_model_save_path():
    return save_dir + '/' + model_name


############################
##         MODEL          ##
############################
scale_fact = 4    # resolution multiplication factor
res_blocks = 3    # amount of residual blocks the network has (+1)


############################
##        TRAINING        ##
############################
# Adjust "crops_p_img", "img_height", "img_width" and "batch_size" to maximize memory usage of GPU.
# "augment_img" will double the amount of pixels calculated below.
# Amount of pixels per batch seen by GPU: "crops_p_img" x "batch_size" x "img_width" x "img_height"
img_width   = 64   # size of the output of the network (play around along with batch_size to maximize memory usage)
img_height  = 64   # this size divided by the scale_fact is the input size of the network
img_depth   = 3    # number of channels (RGB)
epochs      = 15   # amount of times the training data is used
batch_size  = 2    # amount of images to be cropped and fed per batch
verbosity   = 2    # message feedback (0, 1 or 2): higher means more verbose
val_split   = 0.1  # percentage of the dataset to be used for validation
hr_img_path = 'dataset/DIV2K/DIV2K/DIV2K_train_HR/'  # Where the training dataset is.
#hr_img_path = 'pictures/HR/'  # Use this when you want to test the initialization of the filters.
sample_path = 'pictures/HR/'  # Path used for the ModelDiagnoser.


############################
##       EVALUATION       ##
############################
add_callbacks = True      # TensorBoard and visualization/diagnostic functionalities (slows down training)
log_dir       = './logs'  # directory where the Callbacks logs will be stored
tests_path    = 'input/'  # path to the folder containing the HR images to test with

def get_log_path():
    print("Current TensorBoard log directory is: " + log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################
##       PREDICTION       ##
############################
input_width   = 64  # width  size of the input used for prediction
input_height  = 64  # height size of the input used for prediction
overlap       = 16   # amount of overlapped-pixels for predictions to remove the erroneous edges
