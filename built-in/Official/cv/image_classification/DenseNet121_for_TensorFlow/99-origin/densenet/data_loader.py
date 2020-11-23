import numpy as np
from . import preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import os,sys
import numpy as np 
from .train_helper import stage


class DataLoader:

    def __init__(self, config):
        self.config = config   

        #num_training_samples = 1281167
        #num_evaluating_samples = get_num_records(self.eval_filenames)
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames_train = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        num_training_samples = get_num_records(filenames_train)
        self.config['num_training_samples'] = num_training_samples
        #self.config['num_evaluating_samples'] = 50000
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames_val = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
        num_evaluating_samples = get_num_records(filenames_val)
        self.config['num_evaluating_samples'] = num_evaluating_samples
        
        print( 'total num_training_sampels: %d' %  num_training_samples )
        print( 'total num_evaluating_sampels: %d' %  num_evaluating_samples )
        
        self.training_samples_per_rank = num_training_samples
        
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
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox

# since the preprocessing is done here, we add config file
def parse_record(raw_record, is_training, cfg):
    image_buffer, label, bbox = _parse_example_proto(raw_record)
    # set istraining=False to avoid data augement
    # uncomment this for training and evaluating
    #is_training=False
    #is_training=True
    # for 1980 only
    config={'min_object_covered': 0.1, 'aspect_ratio_range': [3. / 4., 4. / 3.], 'area_range': [0.16, 1.0], 'max_attempts': 100}
    if cfg['aug_method'] == 'glu':
      image = preprocessing.parse_and_preprocess_image_record_glu(
          config, image_buffer, height=224, width=224,
          brightness=0.3, contrast=0.6, saturation=0.6, hue=0.13,
          distort=is_training, nsummary=10, increased_aug=False, random_search_aug=False)
    elif cfg['aug_method'] == 'hxb' and cfg['mean'] == 'ori':
        # preprocessing use the simpler version
        image = preprocessing.parse_and_preprocess_image_record_hxb(image_buffer, distort=is_training)
    elif cfg['aug_method'] == 'hxb' and cfg['mean'] == 'me':
        # oppsite due to wrong running order
        image = preprocessing.parse_and_preprocess_image_record_hxb_ori(image_buffer, distort=is_training)
    elif cfg['aug_method'] == 'me':
        image = preprocessing.parse_and_preprocess_image_record_me(image_buffer, bbox, training=is_training)
    else:
        # the ori process function
        image = preprocessing.parse_and_preprocess_image_record(
        config, image_buffer, height=224, width=224,
        brightness=0.3, contrast=0.6, saturation=0.6, hue=0.13,
        distort=is_training, nsummary=10, increased_aug=False, random_search_aug=False)
    #return image, label
    return image, label-1


def make_dataset(config, filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=False, num_threads=10, nsummary=10, shard=False, synthetic=False,
                 increased_aug=False, random_search_aug=False):
    shuffle_buffer_size = 10000
    num_readers = 10
    #num_readers = 1

    if training:
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
    else:
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

    ds = tf.data.Dataset.from_tensor_slices(filenames)

    if not training:
        ds = ds.take(take_count)  # make sure all ranks have the same amount

    if training:
        ds = ds.shuffle(1000)

    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    if training:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))

    ds = ds.map(lambda image, counter: parse_record(image, training, config), num_parallel_calls=14)

    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


