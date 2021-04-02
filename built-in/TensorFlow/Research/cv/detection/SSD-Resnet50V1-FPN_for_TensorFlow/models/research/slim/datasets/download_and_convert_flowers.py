
'Downloads and converts Flowers data to TFRecords of TF-Example protos.\n\nThis module downloads the Flowers data, uncompresses it, reads the files\nthat make up the Flowers data and creates two TFRecord datasets: one for train\nand one for test. Each TFRecord dataset is comprised of a set of TF-Example\nprotocol buffers, each of which contain a single image and label.\n\nThe script should take about a minute to run.\n\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import math
import os
import random
import sys
import tensorflow as tf
from datasets import dataset_utils

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
_NUM_VALIDATION = 350
_RANDOM_SEED = 0
_NUM_SHARDS = 5

class ImageReader(object):
    'Helper class that provides TensorFlow image coding utilities.'

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return (image.shape[0], image.shape[1])

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert (len(image.shape) == 3)
        assert (image.shape[2] == 3)
        return image

def _get_filenames_and_classes(dataset_dir):
    'Returns a list of filenames and inferred class names.\n\n  Args:\n    dataset_dir: A directory containing a set of subdirectories representing\n      class names. Each subdirectory should contain PNG or JPG encoded images.\n\n  Returns:\n    A list of image file paths, relative to `dataset_dir` and the list of\n    subdirectories, representing class names.\n  '
    flower_root = os.path.join(dataset_dir, 'flower_photos')
    directories = []
    class_names = []
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
    return (photo_filenames, sorted(class_names))

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = ('flowers_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS))
    return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    "Converts the given filenames to a TFRecord dataset.\n\n  Args:\n    split_name: The name of the dataset, either 'train' or 'validation'.\n    filenames: A list of absolute paths to png or jpg images.\n    class_names_to_ids: A dictionary from class names (strings) to ids\n      (integers).\n    dataset_dir: The directory where the converted datasets are stored.\n  "
    assert (split_name in ['train', 'validation'])
    num_per_shard = int(math.ceil((len(filenames) / float(_NUM_SHARDS))))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('', config=npu_session_config_init()) as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = (shard_id * num_per_shard)
                    end_ndx = min(((shard_id + 1) * num_per_shard), len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write(('\r>> Converting image %d/%d shard %d' % ((i + 1), len(filenames), shard_id)))
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        (height, width) = image_reader.read_image_dims(sess, image_data)
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def _clean_up_temporary_files(dataset_dir):
    'Removes temporary files used to create the dataset.\n\n  Args:\n    dataset_dir: The directory where the temporary files are stored.\n  '
    filename = _DATA_URL.split('/')[(- 1)]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)
    tmp_dir = os.path.join(dataset_dir, 'flower_photos')
    tf.gfile.DeleteRecursively(tmp_dir)

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
            if (not tf.gfile.Exists(output_filename)):
                return False
    return True

def run(dataset_dir):
    'Runs the download and conversion operation.\n\n  Args:\n    dataset_dir: The dataset directory where the dataset is stored.\n  '
    if (not tf.gfile.Exists(dataset_dir)):
        tf.gfile.MakeDirs(dataset_dir)
    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    (photo_filenames, class_names) = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]
    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Flowers dataset!')
