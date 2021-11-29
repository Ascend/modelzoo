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
import numpy as np
import time
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
import glob
import os
import argparse

input_shape = (224, 224, 3)  # (height, width, channel)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', default=1,
                        help="""batchsize""")
    parser.add_argument('--model_path', default='pb/googlenet_tf_910.pb',
                        help="""pb path""")
    parser.add_argument('--image_path', default = '../../image-50000/',
                        help = """the data path""")
    parser.add_argument('--label_file', default='val_lable.txt',
                        help="""label file""")
    parser.add_argument('--input_tensor_name', default = 'input:0',
                        help = """input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='InceptionV1/Logits/SpatialSqueeze:0',
                        help="""output_tensor_name""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

def read_file(image_name, path):
    with open(path, 'r') as cs:
        rs_list = cs.readlines()
        for name in rs_list:
            if image_name in str(name):
                num = str(name).split(" ")[1]
                break
    return int(num) + 1

class Classifier(object):
    # 定义模型的batch_size规格
    args = parse_args()
    batch_size = int(args.batchsize)

    def __init__(self):
        # --------------------------------------------------------------------------------
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 1）run on Ascend NPU
        custom_op.parameter_map["use_off_line"].b = True
        # 2）recommended use fp16 datatype to obtain better performance
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        # 3）disable remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # 4）set graph_run_mode=0，obtain better performance
        custom_op.parameter_map["graph_run_mode"].i = 0
        # --------------------------------------------------------------------------------

        # load model， set graph input nodes and output nodes
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # create session
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load fronzen graph
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        """
        do infer
        :param image_data:
        :return:
        """
        out_list = []
        total_time = 0
        i = 0
        for data in batch_data:
            t = time.time()
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: data})
            if i > 0:
                total_time = total_time + time.time() - t
            i = i + 1
            out_list.append(out)
        return np.array(out_list), total_time

    def batch_process(self, image_data, label_data):
        """
        batch process
        :return:
        """
        # Get the batch information of the current input data, and automatically adjust the data to the fixed batch
        n_dim = image_data.shape[0]
        batch_size = self.batch_size

        # if data is not enough for the whole batch, you need to complete the data
        m = n_dim % batch_size
        if m < batch_size and m > 0:
            #  Define the Minis that can be divided into several batches
            pad = np.zeros((batch_size - m, 224, 224, 3)).astype(np.float32)
            image_data = np.concatenate((image_data, pad), axis=0)

        # Define the Minis that can be divided into several batches
        mini_batch = []
        mini_label = []
        i = 0
        while i < n_dim:
            # Define the Minis that can be divided into several batches
            mini_batch.append(image_data[i: i + batch_size, :, :, :])
            mini_label.append(label_data[i: i + batch_size])
            i += batch_size

        return mini_batch, mini_label


def normalize(inputs):
    imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    imagenet_mean = tf.expand_dims(tf.expand_dims(imagenet_mean, 0), 0)
    imagenet_std = tf.expand_dims(tf.expand_dims(imagenet_std, 0), 0)
    inputs = inputs - imagenet_mean  # tf.subtract(inputs, imagenet_mean)
    inputs = inputs * (1.0 / imagenet_std)

    return inputs


def image_process(image_path, label_file):
    ###image process
    imagelist = []
    labellist = []
    images_count = 0
    for file in os.listdir(image_path):
        with tf.Session().as_default():
            image_file = os.path.join(image_path, file)
            image_name = image_file.split('/')[-1].split('.')[0]
            image = tf.gfile.FastGFile(image_file, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.central_crop(image, central_fraction=0.875)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [224, 224],align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            img = tf.multiply(image, 2.0)
            images_count = images_count + 1
            if tf.shape(img)[2].eval() == 1:
                img = tf.image.grayscale_to_rgb(img)
            img = img.eval()
            imagelist.append(img)
            tf.reset_default_graph()

            ###labels
            lable = read_file(image_name, label_file)
            labellist.append(lable)
    return np.array(imagelist), np.array(labellist), images_count

def main():
    args = parse_args()
    top1_count = 0
    top5_count = 0

    ###preprocess
    tf.reset_default_graph()
    print("########NOW Start Preprocess!!!#########")
    images, labels, images_count = image_process(args.image_path, args.label_file)

    ###batch
    print("########NOW Start Batch!!!#########")
    classifier = Classifier()
    batch_images, batch_labels= classifier.batch_process(images, labels)

    ###do inference
    print("########NOW Start inference!!!#########")
    batch_logits, total_time = classifier.do_infer(batch_images)

    ###compute accuary
    batchsize = int(args.batchsize)
    total_step = int(images_count / batchsize)
    print("########NOW Start Compute Accuary!!!#########")
    for i in range(total_step):
        top1acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(batch_logits[i], 1), batch_labels[i]), tf.float32))
        top5acc = tf.reduce_sum(tf.cast(tf.nn.in_top_k(batch_logits[i], batch_labels[i], 5), tf.float32))
        with tf.Session().as_default():
            tf.reset_default_graph()
            top1_count += top1acc.eval()
            top5_count += top5acc.eval()
    print('+----------------------------------------+')
    print('the correct num is {}, total num is {}.'.format(top1_count, total_step * batchsize))
    print('Top1 accuracy:', top1_count / (total_step * batchsize) * 100)
    print('Top5 accuracy:', top5_count / (total_step * batchsize) * 100)
    print('images number = ', total_step * batchsize)
    print('images/sec = ', (total_step * batchsize) / total_time)
    print('+----------------------------------------+')

if __name__ == '__main__':
    main()


