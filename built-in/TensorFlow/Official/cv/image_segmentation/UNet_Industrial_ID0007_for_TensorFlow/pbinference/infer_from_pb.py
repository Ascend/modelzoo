#copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import time
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import glob
import os
import sys

from npu_bridge.estimator import npu_ops

from dllogger.logger import LOGGER
import dllogger.logger as dllg

#input_shape = [512, 512, 1]  # (height, width, channel)

# 用户自定义模型路径、输入、输出
model_path='./unet-industrial_tf.pb'
input_tensor_name='input:0'
output_tensor_name='output:0'

class Classifier(object):
    def __init__(self):
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True

        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        custom_op.parameter_map["graph_run_mode"].i = 0

        self.graph = self.__load_model(model_path)
        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        # create session
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_data})
        return out

def DAGM2007_Dataset(data_dir, class_id=1, batch_size=1):
    data_dir = os.path.join(data_dir, "raw_images/private/Class%d" % class_id)

    csv_file = os.path.join(data_dir, "test_list.csv")
    image_dir = os.path.join(data_dir, "Test")
    mask_image_dir = os.path.join(data_dir, "Test/Label")

    input_shape = mask_shape = [512, 512, 1]

    shuffle_buffer_size = 10000

    def decode_csv(line):

        input_image_name, image_mask_name, label = tf.decode_csv(
            line, record_defaults=[[""], [""], [0]], field_delim=','
        )

        def decode_image(filepath, resize_shape, normalize_data_method):
            image_content = tf.read_file(filepath)
            image = tf.image.decode_png(contents=image_content, channels=resize_shape[-1], dtype=tf.uint8)
            image = tf.image.resize_images(
                image,
                size=resize_shape[:2],
                method=tf.image.ResizeMethod.BILINEAR,  # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                align_corners=False,
                preserve_aspect_ratio=True
            )

            image.set_shape(resize_shape)
            image = tf.cast(image, tf.float32)
            if normalize_data_method == "zero_centered":
                image = tf.divide(image, 127.5) - 1
            elif normalize_data_method == "zero_one":
                image = tf.divide(image, 255.0)
            return image

        input_image = decode_image(
            filepath=tf.strings.join([image_dir, input_image_name], separator='/'),
            resize_shape=input_shape,
            normalize_data_method="zero_centered",
        )

        mask_image = tf.cond(
            tf.equal(image_mask_name, ""),
            true_fn=lambda: tf.zeros(mask_shape, dtype=tf.float32),
            false_fn=lambda: decode_image(
                filepath=tf.strings.join([mask_image_dir, image_mask_name], separator='/'),
                resize_shape=mask_shape,
                normalize_data_method="zero_one",
            ),
        )

        label = tf.cast(label, tf.int32)

        return (input_image, mask_image), label

    dataset = tf.data.TextLineDataset(csv_file)
    dataset = dataset.skip(1)  # Skip CSV Header
    dataset = dataset.cache()
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=decode_csv,
            num_parallel_calls=64,
            batch_size=batch_size,
            drop_remainder=True,
        )
    )
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset

def iou_score_fn(y_pred, y_true, threshold, eps=1e-5):
    y_true = y_true > threshold
    y_pred = y_pred > threshold

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = y_true * y_pred
    intersection = tf.reduce_sum(intersection, axis=(1, 2, 3))
    numerator = 2.0 * intersection + eps
    divisor = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3)) + eps
    return tf.reduce_mean(numerator / divisor)

def main():

    filepath = sys.argv[1]

    classifier = Classifier()

    ds = DAGM2007_Dataset(data_dir=filepath, class_id=1, batch_size=2)
    iter = ds.make_initializable_iterator()
    ds_sess = tf.Session()
    ds_sess.run(iter.initializer)
    next_element = iter.get_next()

    eval_metrics = dict()

    IOU_THS = [[],[],[],[],[],[],[],[]] 

    i = 1
    while True:
        try:
            # features
            input = ds_sess.run(next_element)
            batch_data = input[0]
            batch_labels = input[1]

            # input_image, mask_image, labels
            input_image = batch_data[0]
            mask_image = batch_data[1]
            labels = batch_labels

            y_pred = classifier.do_infer(input_image)

            labels = tf.cast(labels, tf.float32)
            labels_preds = tf.reduce_max(y_pred, axis=(1, 2, 3))
            j = 0
            for threshold in [0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:
                tf.reset_default_graph()
                with tf.Session() as eval_sess:
                    iou_score = iou_score_fn(y_pred=y_pred, y_true=mask_image, threshold=threshold)
                    eval_results = eval_sess.run(iou_score)
                    eval_metrics["IoU_THS_%s" % threshold] = tf.metrics.mean(iou_score)

                    IOU_THS[j].append(eval_results) 
                j += 1

            i += 1
            print("======batch %s finished ======" % str(i))

        except tf.errors.OutOfRangeError as e:
            print("### Total IoU_THS_0.05: ",  np.mean(IOU_THS[0]))
            print("### Total IoU_THS_0.125: ", np.mean(IOU_THS[1]))
            print("### Total IoU_THS_0.25: ",  np.mean(IOU_THS[2]))
            print("### Total IoU_THS_0.5: ",   np.mean(IOU_THS[3]))
            print("### Total IoU_THS_0.75: ",  np.mean(IOU_THS[4]))
            print("### Total IoU_THS_0.85: ",  np.mean(IOU_THS[5]))
            print("### Total IoU_THS_0.95: ",  np.mean(IOU_THS[6]))
            print("### Total IoU_THS_0.99: ",  np.mean(IOU_THS[7]))

            break

    ds_sess.close()
    classifier.sess.close()


if __name__ == '__main__':
    main()



