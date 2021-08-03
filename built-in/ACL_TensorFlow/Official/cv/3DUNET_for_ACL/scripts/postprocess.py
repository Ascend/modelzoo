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
import numpy as np
import time
import os
import sys
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from transforms import NormalizeImages, OneHotLabels, apply_transforms, PadXYZ, RandomCrop3D, \
    RandomHorizontalFlip, RandomGammaCorrection, RandomVerticalFlip, RandomBrightnessCorrection, CenterCrop, \
    apply_test_transforms, Cast

from losses import make_loss, eval_dice, total_dice

if __name__ == '__main__':

    resultpath = sys.argv[1]
    label = sys.argv[2]

    ds_sess = tf.Session()
    class_1 = []
    class_2 = []
    class_3 = []
    total = []
    weights = []

    i = 0
    for file in os.listdir(resultpath):
        if file.endswith(".bin"):
            print("start postprocessing",i)
            batch_logits = np.fromfile(resultpath + "davinci_input_" + str(i) + "_output0.bin", dtype='float32').reshape(1,224,224,160,4)
            batch_labels = np.fromfile(label + "label_" + str(i) + ".bin", dtype='float32').reshape(1,224,224,160,4)

            logits = batch_logits
            labels = batch_labels

            labels = tf.cast(labels, tf.float32)
            labels = labels[..., 1:]
            logits = logits[..., 1:]

            eval_acc = eval_dice(y_true=labels, y_pred=tf.round(logits))
            total_eval_acc = total_dice(tf.round(logits), labels)

            eval_acc_1, eval_acc_2, eval_acc_3 = ds_sess.run(eval_acc)
            total_eval_acc = ds_sess.run(total_eval_acc)

            class_1.append(eval_acc_1)
            class_2.append(eval_acc_2)
            class_3.append(eval_acc_3)
            total.append(total_eval_acc)
            weights.append(1)

            TumorCore_avg = np.average(class_1, weights=weights)
            PeritumoralEdema_avg = np.average(class_2, weights=weights)
            EnhancingTumor_avg = np.average(class_3, weights=weights)
            WholeTumor_avg = np.average(total, weights=weights)

            MeanDice_mean = np.average([TumorCore_avg, PeritumoralEdema_avg, EnhancingTumor_avg])

            i += 1
    print("###Total result:", "TumorCore: ", TumorCore_avg, \
                              "PeritumoralEdema: ", PeritumoralEdema_avg, \
                              "EnhancingTumor: ", EnhancingTumor_avg, \
                              "MeanDice: ", MeanDice_mean, \
                              "WholeTumor: ", WholeTumor_avg)

    ds_sess.close()

