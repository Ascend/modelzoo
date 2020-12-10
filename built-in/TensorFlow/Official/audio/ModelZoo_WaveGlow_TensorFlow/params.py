# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    n_fft=1024,
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # train
    lr=6e-4, #1e-3
    train_steps=1000000,
    epochs=100,
    save_model_every=20000,
    gen_test_wave_every=20000,
    gen_file='./data/mel_spect/LJ001-0001.mel',
    logdir_root='./logdir',
    tfrecords_dir="./data/tfrecords/",
    train_files = 'ljs_train',
    # Evaluation tfrecords filename
    eval_file = 'ljs_eval.tfrecords',
    # Test tfrecords filename
    test_file = 'ljs_test.tfrecords',
    decay_steps=5000 ,# 8000,
    sigma=1.0,#0.707, # paper 1.0

    # network
    sample_size=16000,
    batch_size=12,
    upsampling_rate=256, # same as hop_length
    n_flows=12,
    n_group=8,
    n_early_every=4,
    n_early_size=2,

    # local condition conv1d
    lc_conv1d=False, #True
    lc_conv1d_layers=2,
    lc_conv1d_filter_size=5,
    lc_conv1d_filter_num=80,

    # local condition encoding
    lc_encode=False,
    lc_encode_layers=2,
    lc_encode_size=128,

    # upsampling by transposed conv
    transposed_upsampling=True,
    transposed_conv_layers=2,
    transposed_conv_layer1_stride= 16 ,
    transposed_conv_layer2_stride=16,
    transposed_conv_layer1_filter_width= 16*2,#1024 ,#  16*5,  # filter width greater than stride, then could leverage context lc
    transposed_conv_layer2_filter_width=16*2,
    transposed_conv_channels=80,

    # wavenet
    n_layers=8,
    residual_channels=256,
    skip_channels=256,
    kernel_size=3,

)
