import numpy as np
from . import preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import os,sys
import numpy as np 
sys.path.append("..")
from trainers.train_helper import stage

# add vgg processing function
# from . import vgg_preprocessing

class DataLoader:

    def __init__(self, config):
        self.config = config   

        num_training_samples = 1281167
        #num_evaluating_samples = get_num_records(self.eval_filenames)
        self.config['num_training_samples'] = num_training_samples
        self.config['num_evaluating_samples'] = 50000
        print( 'total num_training_sampels: %d' %  num_training_samples )
        
        self.training_samples_per_rank = num_training_samples


    def get_train_input_fn_synthetic(self):
        batch_size = self.config['batch_size']
        input_shape = [self.config['height'], self.config['width'], 3]
        input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
        label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]))
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
        ds = ds.batch(batch_size)
        return ds
        
    def get_train_input_fn(self):
        # filenames = self.train_filenames
        filenames = None
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
        shard = self.config['shard']

        return make_dataset(self.config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=True, num_threads=num_threads, nsummary=10, shard=shard, synthetic=False,
                 increased_aug=increased_aug )

    def get_eval_input_fn(self):
        # filenames = self.eval_filenames
        filenames = None
        # take_count = get_num_records(self.eval_filenames)
        take_count = 50000
        batch_size = self.config['batch_size']
        height = self.config['height']
        width = self.config['width']
        brightness = self.config['brightness']
        contrast = self.config['contrast']
        saturation = self.config['saturation']
        hue = self.config['hue'] 
        num_threads = self.config['num_preproc_threads']
        shard = self.config['shard']

        return make_dataset(self.config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=False, num_threads=num_threads, nsummary=10, shard=shard, synthetic=False,
                 increased_aug=False)

    def get_input_pipeline_op(self, inputs, labels, mode):
        with tf.device('/cpu:0'):
            preload_op, (inputs, labels) = stage([inputs, labels])

        with tf.device('/gpu:0'):
            gpucopy_op, (inputs, labels) = stage([inputs, labels])
        return preload_op, gpucopy_op, inputs, labels

    def normalize_and_format(self, inputs, data_format):

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

def _parse_example_proto(example_serialized):
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox

# since the preprocessing is done here, we add config file
def parse_record(raw_record, is_training, cfg):
  image_buffer, label, bbox = _parse_example_proto(raw_record)
  config={'min_object_covered': 0.1, 'aspect_ratio_range': [3. / 4., 4. / 3.], 'area_range': [0.16, 1.0], 'max_attempts': 100}

  if cfg['aug_method'] == 'hxb':
    # preprocessing use the simpler version
    image = preprocessing.parse_and_preprocess_image_record_hxb(image_buffer, distort=is_training)
  else:
    # the ori process function
    image = preprocessing.parse_and_preprocess_image_record(
      config, image_buffer, height=224, width=224,
      brightness=0.3, contrast=0.6, saturation=0.6, hue=0.13,
      distort=is_training, nsummary=10, increased_aug=False, random_search_aug=False)
  #return image, label
  # label -1 for VGG in ImageNet
  return image, label-1

def read_rawdata(file_path_tensor):
    def _read_file(file_path):
        image = tf.gfile.GFile(file_path, 'rb').read()
        return image
    return tf.py_func(_read_file, inp=[file_path_tensor], Tout=tf.string)

def parse_function(filename, label):
    image = read_rawdata(filename)
    image_decoded = tf.image.decode_jpeg(image, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    label = tf.cast(label, dtype=tf.int32)
    return image_resized, label


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
        ds = ds.batch(batch_size) 
        return ds
    else:
        shuffle_buffer_size = 10000
        num_readers = 1
        # in my env, can NOT find these variables so we set them manually
        rank_size = 1
        rank_id = 0

        if config['data_type'] == 'RAW DATA':
            images = []
            labels = []
            with tf.gfile.GFile(config['label_index_url'], 'r') as f:
                for line in f.readlines():
                    tmp_list = line.strip().split(" ")
                    image_file = os.path.join(config['data_url'], tmp_list[0])
                    images.append(image_file)
                    labels.append(int(tmp_list[-1]))

            ds = tf.data.Dataset.from_tensor_slices((images, labels))
        else:
            if training:
                filename_pattern = os.path.join(config['data_url'], '%s-*')
                filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
            else:
                filename_pattern = os.path.join(config['data_url'], '%s-*')
                filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

            ds = tf.data.Dataset.from_tensor_slices(filenames)

        if shard:
            # split the dataset into parts for each GPU
            ds = ds.shard(rank_size, rank_id)

        if not training:
            ds = ds.take(take_count)  # make sure all ranks have the same amount

        # comment this to fix data reader
        if training:
            #ds = ds.shuffle(1000, seed=7 * (1 + rank_id))
            # for gpu, we use random see
            ds = ds.shuffle(1000)

        if config['data_type'] == 'TFRECORD':
            ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
            counter = tf.data.Dataset.range(sys.maxsize)
            ds = tf.data.Dataset.zip((ds, counter))

        if training:
            # comment this to fix order
            # the same reason for gpu
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))

        if config['data_type'] == 'RAW DATA':
            ds = ds.map(lambda image, label: parse_function(image, label), num_parallel_calls=48)
        else:
            ds = ds.map(lambda image, counter: parse_record(image, training, config), num_parallel_calls=48)

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size= tf.contrib.data.AUTOTUNE)
        return ds


