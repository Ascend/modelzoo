import numpy as np
from . import preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import horovod.tensorflow as hvd
import os,sys
import numpy as np 
sys.path.append("..")
from trainers.train_helper import stage

from models.mlperf_compliance import mlperf_log


class DataLoader:

    def __init__(self, config):
        self.config = config   

        if config['data_dir']:
            filename_pattern = os.path.join(self.config['data_dir'], '%s-*')
            self.train_filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
            self.eval_filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
            num_training_samples = get_num_records(self.train_filenames)
            num_evaluating_samples = get_num_records(self.eval_filenames)

            if self.config['print_mlperf_log']:
                mlperf_log.resnet_print(key=mlperf_log.INPUT_ORDER)
                mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES, value=num_training_samples)
                mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES, value=num_evaluating_samples)

            self.config['num_training_samples'] = num_training_samples 
            self.config['num_evaluating_samples'] = num_evaluating_samples 

            if hvd.rank() == 0:
                print( 'total num_training_sampels: %d' %  num_training_samples )
        else:
            raise ValueError('data_dir missing. Please pass --synthetic if you want to run on synthetic data. Else please pass --data_dir')
        
        self.training_samples_per_rank = num_training_samples // hvd.size()


    def get_train_input_fn_synthetic(self):
        input_shape = [self.config['height'], self.config['width'], 3]
        input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
        label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]))
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
        ds = ds.batch(batch_size)
        return ds
        
    def get_train_input_fn(self):
        filenames = self.train_filenames
        take_count = self.training_samples_per_rank
        batch_size = self.config['batch_size']
        height = self.config['height']
        width = self.config['width']
        brightness = self.config['brightness']
        contrast = self.config['contrast']
        saturation = self.config['saturation']
        hue = self.config['hue']
        num_threads = self.config['num_preproc_threads']
        increased_aug = self.config['increased_aug']
        random_search_aug = self.config['random_search_aug']

        return make_dataset(self.config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=True, num_threads=num_threads, nsummary=10, shard=True, synthetic=False,
                 increased_aug=increased_aug, random_search_aug=random_search_aug )

    def get_eval_input_fn(self):
        filenames = self.eval_filenames
        take_count = get_num_records(self.eval_filenames)
        batch_size = self.config['batch_size']
        height = self.config['height']
        width = self.config['width']
        brightness = self.config['brightness']
        contrast = self.config['contrast']
        saturation = self.config['saturation']
        hue = self.config['hue'] 
        num_threads = self.config['num_preproc_threads']

        return make_dataset(self.config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=False, num_threads=num_threads, nsummary=10, shard=True, synthetic=False,
                 increased_aug=False)

    def get_input_pipeline_op(self, inputs, labels, mode):
        with tf.device('/cpu:0'):
            preload_op, (inputs, labels) = stage([inputs, labels])

        with tf.device('/gpu:0'):
            gpucopy_op, (inputs, labels) = stage([inputs, labels])
        return preload_op, gpucopy_op, inputs, labels

    def normalize_and_format(self, inputs, data_format):
        if self.config['print_mlperf_log']:
            imagenet_mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)             #-----------------hwp---------
            imagenet_std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
        else:
            imagenet_mean = np.array([121, 115, 100], dtype=np.float32)
            imagenet_std = np.array([70, 68, 71], dtype=np.float32)
        inputs = tf.subtract(inputs, imagenet_mean)
        inputs = tf.multiply(inputs, 1. / imagenet_std)
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        return inputs




#-------------------------------- Funcs -----------------------------------
def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count

    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


def make_dataset(config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=False, num_threads=10, nsummary=10, shard=False, synthetic=False,
                 increased_aug=False, random_search_aug=False):
    if synthetic and training:
        input_shape = [height, width, 3]
        input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
        label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]))
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
    else:
        shuffle_buffer_size = 10000 
        num_readers = 1
        if hvd.size() > len(filenames):
            assert (hvd.size() % len(filenames)) == 0
            filenames = filenames * (hvd.size() / len(filenames))
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        if shard:
            # split the dataset into parts for each GPU
            ds = ds.shard(hvd.size(), hvd.rank())

        if not training:
            ds = ds.take(take_count)  # make sure all ranks have the same amount

        if training:
            ds = ds.shuffle(1000, seed=7 * (1 + hvd.rank()))

        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
        counter = tf.data.Dataset.range(sys.maxsize)
        ds = tf.data.Dataset.zip((ds, counter)) 
        preproc_func = lambda record, counter_: preprocessing.parse_and_preprocess_image_record(config, 
            record, counter_, height, width, brightness, contrast, saturation, hue,
            distort=training, nsummary=nsummary if training else 0, increased_aug=increased_aug, random_search_aug=random_search_aug)
        ds = ds.map(preproc_func, num_parallel_calls=num_threads)
        if training:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, seed=5*(1+hvd.rank())))
    ds = ds.batch(batch_size)
    # ds = ds.prefetch(10000)
#    ds = ds.apply( tf.contrib.data.prefetch_to_device('/gpu:0', buffer_size=10000) )
    return ds

