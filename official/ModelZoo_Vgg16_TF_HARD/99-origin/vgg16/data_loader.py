import numpy as np
from . import preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import os,sys
import numpy as np
import horovod.tensorflow as hvd


class DataLoader:

    def __init__(self, config):
        self.config = config   

        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames_train = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        self.num_training_samples = get_num_records(filenames_train)
        self.config['num_training_samples'] = self.num_training_samples

        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames_val = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
        self.num_evaluating_samples = get_num_records(filenames_val)
        self.config['num_evaluating_samples'] = self.num_evaluating_samples
        
        print( 'total num_training_sampels: %d' %  self.num_training_samples )
        print( 'total num_evaluating_sampels: %d' %  self.num_evaluating_samples )
        
        self.training_samples_per_rank = self.num_training_samples
        
    def get_train_input_fn(self):
        take_count = self.training_samples_per_rank
        batch_size = self.config['batch_size']
        shard = self.config['shard']

        return make_dataset(self.config, take_count, batch_size,
                   training=True, shard=shard)

    def get_eval_input_fn(self):
        take_count = self.num_evaluating_samples
        batch_size = self.config['batch_size']
        shard = self.config['shard']

        return make_dataset(self.config, take_count, batch_size,
                 training=False, shard=shard)


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

    image = preprocessing.parse_and_preprocess_image_record(image_buffer, bbox, training=is_training)

    # label-1 for VGG in slimImageet dataset
    return image, label-1


def make_dataset(config, take_count, batch_size,
                 training=False, shard=False):

    shuffle_buffer_size = 10000
    num_readers = 10

    rank_size = hvd.size()
    rank_id = hvd.local_rank()

    if training:
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
    else:
        filename_pattern = os.path.join(config['data_url'], '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

    ds = tf.data.Dataset.from_tensor_slices(filenames)

    if shard:
        ds = ds.shard(rank_size, rank_id)

    if not training:
        ds = ds.take(take_count)  # make sure all ranks have the same amount

    if training:
        ds = ds.shuffle(1000, seed=7*(1+rank_id))

    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    if training:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, seed=5*(1+rank_id)))

    ds = ds.map(lambda image, counter: parse_record(image, training, config), num_parallel_calls=14)

    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


